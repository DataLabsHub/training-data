"""
Within tile prefix scan is the problem — a true prefix over TILE with TILE threads in parallel requires a scan algorithm. In shared memory, a Hillis-Steele scan is fine and deterministic. Let me use TILE = BLOCK elements per tile but process with the block's threads.

Actually, let me reconsider. This per-expert approach re-reads flat_e E times (each CTA reads all n). For E=512, n=24576: total reads 12.6M. But also each CTA is mostly idle (only ~n/E ~ 48 matches typical, but skew). It works but reads a lot.

**Better: single kernel, one CTA per flat-id-tile, handles ALL experts.** Each CTA processes a contiguous tile of flat-ids. It computes per-tile per-expert counts, then atomically increments a global running counter per expert to get the base offset... but atomics break stability across tiles.

Unless: I process tiles in order (they launch in grid order...) — no guarantees.

I keep hitting the stability wall. The fundamental issue: stable counting sort requires a global ordered scan.

**Resolution**: Use a **two-stage approach where stage 1 computes per-tile-per-expert counts (deterministic), stage 2 computes per-tile offsets via prefix over tiles (deterministic, since tile order is fixed), stage 3 does per-tile within-tile scan.**

Let me just implement it fully. Steps (all on device):

1. `cnt_tile[tile][e]`: for each flat-id-tile, count per expert. Grid over tiles. n small → few tiles.
2. `tile_off[tile][e]` = prefix over tiles of cnt_tile, then padded. Combine with block padding.
3. `rank` within tile: prefix scan within tile (deterministic).
4. destination = seg_padded_start[e] + prefix(tile contribution) + within_tile_rank.

Wait, but I also need the block-padding conversion: the padded segment per expert is `nblocks_e * BLOCK`. The rank within expert (0-based without padding) maps to `seg_start[e] + rank`. And seg_start[e] = sum of padded lengths of experts < e. This requires final counts first.

This is a genuine multi-pass counting sort. Getting complicated but let me simplify by choosing tile = whole thing sized to fit.

Given n ≤ 32768, I could fit one CTA to process ALL flat-ids? 32768 elements can't fit in one CTA's shared (needs per-expert...). But I can do a **grid-stride with a global serial scan in shared**... no.

Let me reconsider using a totally different, simpler determinism: **Do the count and the rank-assignment with a "radix" where I use per-expert atomic counters, but assign rank atomically, and THEN fix ordering by noting the rank gives each flat-id a destination; the set of destinations per expert is exactly {0..count-1} but possibly permuted. To restore stability, I need ascending flat-id = ascending rank.**

What if I process using atomics but in a way where I ALSO track... no.

**Pragmatic final decision**: Given n is small (≤32768) and this is launch-bound, I'll implement the **O(n * num_nonempty_experts) "scan per expert"** approach BUT with the per-expert CTA doing an in-kernel sort-free ordered gather via a **serial scan over n with vectorized reads and a shared-memory output staging**. To keep it acceptable, I'll have each CTA (per expert) scan n flat-ids in tiles, and within each tile do a **parallel prefix-scan using cooperative shared-memory Hillis-Steele** for the matches, giving O(n/E) effective... no, O(n) per CTA.

Hold on. Let me reconsider the actual total work. Per-expert CTA scanning all n flat-ids: E CTAs × (n loads) = E·n loads. Worst 512×24576 = 12.6M loads. Combined with each CTA having 256 threads doing the scan cooperatively, the scan itself is n steps but 256-way... The tile-based prefix makes a tile of 256 processed in O(256) shared scan using Hillis-Steele O(256 log 256). Per tile O(256*8)=2048 shared ops for 256 elements. For 24576 elements/tile-count = 96 tiles, 96*2048 = 196K shared ops per CTA. × 512 CTAs / 108 SMs... ~930K shared ops per SM. Shared ops are fast. Should be fine, tens of microseconds.

But reading flat_e E times from global is the main cost. 12.6M × 4 bytes = 50MB at ~1.5TB/s = ~33us. Hmm.

Actually — I realize I can do much better with **one CTA per flat-id-tile handling all experts**, and get stability by processing the tiles in a **deterministic order using a global scan of per-tile-per-expert counts**. Let me just commit to the full 4-pass counting sort since it's the proper O(n) algorithm and really is the "algorithmic reformulation" the task wants.

Let me design it cleanly with fixed parameters:

Pass 1 (count per tile per expert): Grid over tiles (each tile = TILE=1024 flat-ids). Each CTA atomically adds to... let me use per-tile counts in a 2D array `tile_cnt[tile * E + e]`. But compute in shared then write. Tiles count = ceil(n/1024) ≤ 32. E ≤ 512. So tile_cnt is ≤ 32*512 = 16384 ints = 64KB. OK.

Actually simpler: since I need a global prefix over BOTH expert dimension and tile position... This is a stable counting sort. Let me think about what really determines output.

Output layout: experts ascending. Within expert e, the `count_e` flat-ids in ascending flat-id order, then padding.

The destination of flat-id f (expert e, rank r within e) = `pre_padded[e] + r` where `pre_padded[e] = sum_{e'<e} padded(e')`, and `padded(e) = ceil(count_e/BLOCK)*BLOCK`. And rank r = #{f' < f : expert[f']==e}.

So I need: (a) count_e for all e, (b) padded prefix, (c) rank r for each f. Steps:

1. **count_e**: atomicAdd over flat-ids. (order-independent, deterministic result)
2. **padded & seg_start**: computed from count_e via a small scan (single CTA, E threads).
3. **rank_r**: #{f'<f: expert[f']==e}. This is the stable rank. 

For (3), the cleanest deterministic parallel method given small n: **tile-then-scan**:
   - 3a. For each tile t and expert e: `tile_cnt[t][e]` = count of e in tile t. (grid over tiles, deterministic)
   - 3b. For each tile t and expert e: `tile_base[t][e]` = sum of tile_cnt[t'][e] for t' < t. (prefix over tiles, per expert) — small, done in kernel.
   - 3c. For each flat-id f in tile t: within-tile rank = #{f' in tile t, f'<f, expert[f']==e}. Compute with shared-memory prefix scan in the tile's CTA. Deterministic.
   - destination = seg_start[e] + tile_base[t][e] + within_tile_rank.

This is O(n) total reads across all passes, plus small scan overheads. Efficient.

But implementing 3 passes + the within-tile scan in Triton is hard. In raw CUDA, all doable but lengthy.

Hmm. Given the effort, let me reconsider if simpler approach suffices for "optimized". 

Let me reconsider the magnitude of the naive reference. The naive loop: for each expert with count>0, it does `torch.nonzero`, `torch.full`, `idx.to`, `flat_weights[idx]`, slice-append, `torch.full`, plus the final `torch.cat` of up to E chunks. Each of these is a kernel launch + host-side Python/aten sync. For E=512 (case 7 has E=512, T=4096,TOPK=2,n=8192), the loop runs ~512 times (assuming most experts non-empty, but stride 3 silences 1/3 → ~341 non-empty). Each iteration ~8 kernel launches → ~2700 kernel launches, each with a host sync. At ~2-5us per launch overhead → ~5-15ms. Plus the per-expert `nonzero` is O(n) each → O(E*n) = 341*8192 = 2.8M... fine.

So the naive is on the order of milliseconds. My kernel just needs to beat that and be fast in absolute terms. A ~30-100us kernel is a massive win (50-150x).

Given that, let me go with the **clean per-expert CTA approach** but optimize it to avoid re-reading n times per expert by... actually you know what, let me do the O(n) multi-pass properly but keep it as simple as possible.

Actually, simplest correct O(n)-ish approach that avoids the stability headache entirely, using **one CTA processes a tile, computes within-tile ranks with shared scan, and gets cross-tile base via a deterministic global per-tile offset that I compute in a prior tiny kernel**. Let me structure:

Actually, let me reconsider ONE more time for the truly simplest correct parallel thing:

**Since n ≤ 32768, just do a serial scan in a single CTA with 1024 threads** cooperatively exposing parallelism via the prefix scan across a shared array of per-expert... 

You know, I'll go with per-expert CTA. It's simple, clearly correct, deterministic (scan in ascending flat-id order), and the cost is bounded. Let me optimize the per-CTA scan to be sequential but vectorized, and accept the E*n reads but make them coalesced.

Correction — actually I realize per-expert CTA has a naive-scan cost of n sequential "steps" per CTA (even with 256 threads cooperating on the tile scan). To process 24576 elements, the CTA loops over 96 tiles, and per tile does shared prefix scan ~ O(256 log 256). But the sequential dependency across tiles (running count) means the tiles are sequential. 96 sequential tiles, each with a shared scan — the shared scan on 256 elements with Hillis-Steele is 8 steps. So 96*8 = 768 "steps" of shared work, plus global loads. Each CTA reads n=24576 elements. With 256 threads loading 96 tiles of 256 = coalesced. That's fine, ~fast. Memory: 512 CTAs × 24576 × 4B = 50MB. ~33us. Plus the writes and small per-expert work.

But wait — most experts are nearly empty (power law). 512 CTAs where most match 0-50 elements but each still scans all 24576 elements. That's wasteful but bounded. And with 512 CTAs on 108 SMs, ~5 waves.

Alternatively, better load balancing and less waste: **one CTA per flat-id-tile, handling all experts, with the cross-tile determinism handled by a global scan.** Let me just commit to this proper design because it's the "algorithmic reformulation" and genuinely O(n).

Final proper design (CUDA, load_inline):

Let me define TILE = 1024 (flat-ids per tile). NT = ceil(n / TILE) ≤ 32 for n≤32768.

**Kernel 1: `count_tiles`** — grid = NT CTAs. Each CTA loads its tile of flat-ids into shared, computes per-expert counts (each thread atomically adds to shared counts array of size E; E≤512 ints, shared is 2KB, fits). Writes `tile_cnt[tile*E + e]`. Also compute within-tile ranks later.

Hmm, this needs E-sized shared per CTA, 512*4=2KB fine. But atomics in shared per expert, and E up to 512 with only 256 threads — each thread processes multiple elements doing atomicAdd on shared counts. Deterministic count (just the values). OK.

**Kernel 2: `build_layout`** — single CTA, E threads (or 1 block with E threads). Reads tile_cnt[NT*E], computes:
   - `count[e] = sum over tiles` 
   - `nblocks[e] = ceil(count/BLOCK)`, `padded[e]`, `seg_start[e]` (prefix of padded)
   - `total_padded`
   - Also, for each (tile, expert): `tile_base[tile][e] = prefix over tiles < tile of tile_cnt` (deterministic, tile order fixed). Wait — but the destination for each flat-id = seg_start[e] + (number of matches in earlier tiles for e) + (within-tile rank). Yes `tile_base[tile][e] = sum_{t'<t} tile_cnt[t'][e]`. Compute and store in a small device array `tile_base[NT*E]`. But careful: the within-tile rank and tile_base give the rank r = #{f'<f in same expert}. Correct.
   Wait, this maps flat-id f to seg_start[e] + r where r is its rank among ALL matching flat-ids in ascending order. Since tiles are in flat-id order (tile t = flat ids [t*TILE, (t+1)*TILE)), prefix over earlier tiles + within-tile prefix (also ascending) = ascending global rank. Deterministic and stable. 
   - Also emit `block_expert` via repeat_interleave: for blocks, `block_expert[i]` = expert of block i. Compute from nblocks. Launch grid over total_blocks.
   - Also num_tokens_post_pad.

**Kernel 3: `scatter`** — grid = NT CTAs (or one big grid over flat-ids). For each flat-id f: expert e, tile t, within-tile rank (via shared prefix scan in tile), dest = seg_start[e] + tile_base[t][e] + within_rank. Write out_ids[dest]=f, out_w[dest]=flat_w[f]. Also pre-zero out_ids to sentinel and out_w to 0 (or fill in this kernel by also writing padding... easier separate memset or init).

This is the proper counting sort. 

Complexity of implementing within-tile rank with shared prefix scan for E up to 512 experts and TILE=1024 — messy in CUDA but doable. Actually within a tile, I need per-position: rank = number of preceding positions in tile with SAME expert. That's a segmented scan per expert within the tile. 

Hmm. Given E up to 512 and TILE 1024, I can compute within-tile ranks as: for each element i in tile, its within-tile rank = sum_{j<i, same expert}. To compute in shared: allocate shared `rank[TILE]` and `cnts_shared[TILE]`. Method: 
- Each thread i: cnts_shared[i] = (expert[i] == ... ) no, expert varies.

This is a **segment-scan** which in shared can be done with a Hillis-Steele but segments break the simple addition... Actually no! I can do a **stable counting within tile** using a different structure. For each position i with expert e_i, within-rank = count of j<i in tile with expert==e_i. 

Simplest: for each element i (thread i), loop j over tile positions < i counting matches. That's O(TILE²) per tile = 1024² = 1M per tile × NT=32 tiles = 33M. Coordinated... ~fast enough? 33M int compares, fine.

Even simpler: since I'm already going to compute `tile_cnt`, I can reuse. But within-tile rank per element is inherently a scan.

Given TILE=1024, per-tile O(TILE²)=1M ops, 32 tiles=33M ops over 108 SMs ≈ 300K ops/SM ≈ fast. This is fine! Let me use this simple quadratic-within-tile.

So within CTA for a tile, thread i computes within_rank by scanning previous tile positions. = O(TILE²) per tile, TILE=1024 → 1M shared reads per tile CTA. Acceptable.

Hmm wait, but I can simplify MUCH more. Since I'm doing within-tile O(TILE²) anyway, why not just do the WHOLE thing per-expert with O(n) per expert but no... no, O(n²) with n=24576 is 600M, worse than tile approach's 33M. The tile approach is good.

Actually, let me simplify further. Let me merge kernel 3's within-tile rank with kernel coupling... The tiles: instead of a shared prefix, note within-tile-rank for position i = count of j in [tile_start, i) with expert[j]==expert[i]. I'll compute with thread i looping. Good.

Let me also reconsider: do I even need `tile_cnt` separately? I need `count[e]` (for padding) and `tile_base[t][e]`. I can get count[e] = sum over tiles of tile_cnt, and tile_base from prefix. Both from tile_cnt. Or get count[e] first via one kernel then anything. Let me combine kernel 1 and 2 partially: kernel 1 computes tile_cnt. Kernel 2 (small, single CTA) computes count, nblocks, padded, seg_start, total. Kernel 3 (small, single CTA) computes tile_base from tile_cnt (prefix over tiles per expert). Kernel 4 scatters. Plus kernel for block_expert and num_post.

That's 4-5 kernels. For this launch-bound task with tiny n, launch overhead ~5×5us = 25us. Hmm, that's not tiny. But the naive is ms. And fused into fewer kernels better.

Let me reduce kernel count:
- **Kernel A** (grid NT, TILE threads): compute tile_cnt (per-tile per-expert counts) AND within-tile ranks (per element, stored to `within_rank[flat_id]` or computed on the fly). Store `within_rank[i]` per flat-id to global arr `rank[i]`. This kernel also could compute tile_base? No, needs cross-tile.
- Actually within-tile rank can be computed in kernel A and stored. Then kernel for layout.
- **Kernel B** (single CTA): from tile_cnt compute count[e], nblocks, padded, seg_start, total, and tile_base[tile][e]. Also build block_expert here if small or in kernel C. Emit num_post.
- **Kernel C** (grid over flat-ids): for each flat-id f: dest = seg_start[e] + tile_base[t][e] + rank[f]; write. Also need to zero/sentinel-initialize out arrays first.

So 3 kernels + init. Total ~ maybe 30us. Good enough, dramatically beats naive.

Can I fuse more? Kernel A and C could be one if I store rank[f] and then... no, need seg_start across experts which is global. Keep 3.

Alternatively, single kernel that does everything with atomics for cross-tile determinism is impossible (stability).

Let me just go with 3 kernels. Actually, let me reconsider whether the cross-tile determinism computation is worth it vs per-expert. The per-expert approach is ONE kernel (grid = E), simpler, and O(E·n)/TILE_coop. But re-reads n per expert. 3-kernel counting sort is cleaner memory-wise but more kernels.

Given the task emphasizes "algorithmic_reformulation" and beating torch.compile — a compact counting sort is the right story. Let me implement the 3-kernel version. But I worry about implementation bugs in raw CUDA.

Let me reconsider doing it in Triton for cleaner correctness. Triton can do atomics (tl.atomic_add on global). But the cross-tile determinism needs multiple passes with global scans — Triton can do reductions with tl.atomic or tl.reduce, but complex cross-CTA coordination in Triton is hard. Triton is per-CTA; coordinating a global prefix over tiles across CTAs within one kernel is not straightforward (no global sync). So multi-kernel with Triton: pass intermediates to global memory, launch new kernels. Doable but verbose.

I'll write it in raw CUDA via `load_inline`. Let me carefully implement.

Actually, let me reconsider the whole approach once more for simplicity and minimum bug risk, given I can't test in this environment (no tools). I must be confident it's correct.

Let me choose the **per-expert single-kernel** approach — it's the simplest correct one and I can reason about it fully. Cost O(E·n) reads worst case ~50MB and each CTA does a serial-but-vectorized scan. Let me actually bound it better: per-expert CTA with the O(n) scan but with within-tile prefix to parallelize.

Wait, per-expert CTA within-tile ordered gather: CTA for expert e, 256 threads, scans tile of 256 flat-ids. For each element i in tile, is it expert e? Maintain running count across the whole scan (global for the CTA). Need ordered write: element i with expert e should go to dest = base + (running count before i). With 256 threads processing a tile in parallel, I need within-tile prefix of matches. Hillis-Steele shared scan over `match[i]` gives prefix; then each match writes to base + prefix. Cross-tile: base increments by tile's match count between tiles (sequential dependency, but tiles processed sequentially by the CTA — fine).

Hillis-Steele scan over 256 elements in shared: 8 steps. Per tile ~8 steps + loads. 96 tiles for n=24576 → 768 steps + loads per CTA × 512 CTAs. It's fine.

But it reads flat_e n times per expert (E× total). And only writes matches. The wasteful E× read of n... 

OK you know what, both approaches work. Let me go with the **counting-sort 3-kernel** because it's genuinely O(n) reads and is the "right" algorithmic reformulation, and it's what demonstrates algorithmic skill. I'll be very careful with the CUDA code.

Hold on, let me reconsider complexity of the counting sort in CUDA and reduce kernels / simplify.

Let me reconsider: maybe I'm overengineering. Let me reconsider the simplest thing that's still O(n) and single-pass deterministic:

**Insight**: If I process flat-ids in ascending order within a SINGLE CTA, and that CTA covers all n elements, stability is trivial with a serial scan but that's n sequential steps. n=24576 sequential steps by 1024 threads cooperating... Use the tile approach WITHIN one CTA but tiles are processed by... no.

Actually, **one big CTA with 1024 threads, tile = 1024, and I want a GLOBAL stable scan across all NT tiles**. If one CTA processes all 32 tiles sequentially with per-tile prefix scans (Hillis-Steele per tile) and a running per-expert base in shared — I get global stable order! Because the CTA processes tiles in ascending order and maintains running per-expert counts in shared memory across tiles. Within tile, Hillis-Steele prefix on `(expert[i])` won't give per-expert prefix though (Hillis-Steele adds all, not per-expert)...

Hmm, within a tile, per-element within-tile rank needs per-expert prefix. With one CTA and 1024 elements, thread i could compute within-tile rank by scanning j<i — O(TILE²)=1M per tile, ×32 = 33M shared ops in one CTA. That's a lot for one SM but 33M shared ops ~ maybe 33M cycles/... no, 1024 threads in parallel, 1024 steps each (scan 1024 elements) → 1024 steps × some, ~fast. But one CTA = one SM underutilized.

This is getting too deep. Let me just settle: **3-kernel counting sort, within-tile ranks via per-element scan.** I'll write it carefully. n≤32768 so TILE=1024, NT≤32.

Let me now write the CUDA kernels precisely.

Definitions:
- n = T*TOPK, flat_e (int32→treat as int32 for compare, but expert ids in [0,E)), flat_w (fp32).
- E = num_experts, BLOCK.
- TILE = 1024. NT = (n + TILE-1)/TILE.
- sentinel = n.

Outputs: out_ids (int32, total_padded), out_w (fp32, total_padded), block_expert (int32, total_blocks), num_post (int32 [1]).
- total_blocks = sum_e nblocks[e], total_padded = total_blocks*BLOCK.

**Pass 1 kernel — `tile_count_and_rank`**: grid = (NT,), block = TILE threads (1024).
Each block t handles flat ids [t*TILE, min((t+1)*TILE, n)).
- Shared: `s_cnt[E]` = per-expert counts in this tile. E ≤ 512, shared 512*4=2KB. Initialize to 0.
- Load flat ids: for i in 0..TILE (but only valid if < n), `e = flat_e[base+i]`; atomicAdd(&s_cnt[e], 1).
  But careful: masked positions (>= n) should not count. In this kernel positions i >= n are invalid; don't add. But shared s_cnt indexed by e valid (0<=e<E).
- Compute within-tile rank for each valid i: `wr[i] = #{j in [0,i) : flat_e[base+j]==e_i}`. This requires scanning. Each thread i does a loop over j in [0,i). O(TILE²) shared reads. Tile 1024 → 1024*1024/2 = 500K reads per thread-block... wait per thread it's up to 1024 iterations; ×1024 threads = 1M shared reads per tile in parallel (each thread reads up to 1024 shared). Actually the loop is over j in [0,i) for thread i, total O(TILE²)=1M shared accesses per tile but across 1024 threads each doing ~500 — parallel. ~500 sequential shared reads per thread, each ~few cycles. Fine.
  To avoid re-reading global for the within-scan, store the tile's expert ids in shared first: `s_e[i]=flat_e[base+i]`. Then within-rank scans s_e.
- Store to global: `tile_cnt[t*E + e] = s_cnt[e]` (for all e) — write by e-th thread.
- Store `inv_rank[base+i] = wr[i]` for valid i. (A global array, int32, size n.)
  Actually I can store rank later. Let me store `within_rank[flat_id]`.

Hmm, but do I need within_rank stored globally, or recomputed in scatter? Store it to avoid recompute. Global size n ints.

Wait, but there's indexing subtlety: tile t covers [t*TILE, t*TILE+TILE). For i-th position, flat_id = t*TILE + i. Within_rank computed for that flat_id.

**Pass 2 kernel — `build_layout`**: single CTA with E threads (or block of E, E≤512; use blockDim 512, E threads active).
- Read tile_cnt[t*E+e] for t in 0..NT-1.
- Compute count[e] = sum_t tile_cnt. 
- nblocks[e] = (count[e]+BLOCK-1)/BLOCK (integer div; count and block positive; if count 0, nblocks 0). Actually use ceil div.
- padded[e] = nblocks[e]*BLOCK.
- seg_start[e] = prefix sum of padded over e < e. (cumulative)
- total = sum all padded = seg_start[E-1] + padded[E-1]; store num_post[0]=total.
- Store nblocks[e], seg_start[e], padded maybe summed.
- Also compute tile_base[t][e] = sum_{t'<t} tile_cnt[t'][e]. Store in global `tile_base[t*E+e]`. This is a prefix over tiles per expert. Compute per (t,e): thread e reads NT values and prefix. Or restructure: thread e loops t computing prefix. NT≤32, E threads. Each thread e: base=0; for t in 0..NT-1: store tile_base[t*E+e]=base; base+=tile_cnt[t*E+e]. 
- Also emit block_expert: for each block index b, expert e where cumulative nblocks positions. `block_expert[b]` = the e such that block falls in expert e's block range. Since experts ascending and blocks sequential, block_expert = repeat_interleave(arange(E), nblocks). Compute: thread b (blockDim = total_blocks maybe up to... total_padded/BLOCK ≤ n/BLOCK + E ~ small) writes block_expert[b]. But total_blocks unknown until computed. Easier: launch separate small grid or do in pass 2 with E threads: for each e, for k in 0..nblocks[e]-1: block_expert[seg_start[e]/BLOCK + k] = e. seg_start[e] is multiple of BLOCK (since padded multiples of BLOCK). Good. Write block_expert.

**Pass 3 kernel — `scatter`**: grid NT (or over all flat-ids). For flat_id f in tile t:
- e = flat_e[f]; within_rank = within_rank[f].
- dest = seg_start[e] + tile_base[t*E+e] + within_rank.
- out_ids[dest] = f; out_w[dest] = flat_w[f].
Need out arrays initialized to sentinel and 0 first. I'll memset out_ids to sentinel and out_w to 0 in a small init kernel or using cudaMemset. Since out_ids must be int32 sentinel pattern and out_w zero — do a fill kernel or use torch operations? The task says NO non-optimized torch glue ops; but initializing output tensors with torch.full outside kernel... "you must implement everything in-kernel". Hmm. Better fill within kernels.

Actually pass 2 knows total_padded. I can have pass 2 or a tiny kernel fill out_ids[0..total]=sentinel and out_w=0 before scatter. Or scatter kernel's grid covers all padded slots: but scatter handles only non-pad. Let me add the fill into the scatter kernel: grid over padded slots, each writes sentinel/0, but that's total_padded ≤ n + E*BLOCK. Hmm total_padded can be up to ~n + E*(BLOCK-1) roughly, for n=32768, BLOCK=256, E=512 → total up to 32768 + 512*255 = 163K. Grid over 163K threads writing sentinel. Fine. But then scatter also writes to same array — need ordering (fill before scatter). Separate kernel then.

Let me just do: **Pass 2a (init)** fills out arrays; or fold init into pass 2 using its single CTA (thread-idx over total_padded... total up to 163K, single CTA 1024 threads loop). Could be slow-ish but fine. Better: a separate grid-init kernel.

Let me count kernels: (1) tile_count_and_rank, (2) build_layout (also block_expert, num_post), (2b) init_out (grid over total), (3) scatter. Plus output allocation. That's 5 kernel launches + allocs. Launch overhead ~25-30us. Hmm.

Can I combine init_out with kernel 1 or 2? Kernel 1 grid=NT, I could also have it zero out... but total unknown in kernel 1. Put init at start of kernel 2's CTA? Single CTA of total_padded init loop. For total=163K, 1024 threads → 160 iterations each. ~ fine, few us. But I'd need total before init — computed early in kernel 2. So kernel 2 does: compute layout, then init out arrays (each thread loops over total starting at... but total_padded could be 0 if all experts empty; handle). Then how does kernel 3 (scatter) know kernel 2 finished (global sync between kernels is guaranteed by separate launches). Yes, sequential kernels have implicit sync. Good.

So kernels: (1) tile_count_and_rank, (2) build_layout + init_out + block_expert + num_post, (3) scatter. 3 kernels. 

Edge cases:
- total_padded = 0 (all experts empty): out arrays empty, block_expert empty. In practice make_inputs ensures non-empty mostly but could be all empty? With power-law and n≥T*TOPK≥... case 0 T=1,TOPK=1,n=1, some expert gets it, non-empty. Should be fine but handle total=0 (skip init loop, num_post=0).
- Sentinel = n. out_ids padded = n.
- Weights padded = 0.0.
- within_rank for positions >= n (invalid) — not used.
- tile_cnt for invalid positions: I must ensure invalid (base+i >= n) positions don't add to s_cnt. When loading s_e[i], set to a sentinel expert value (like -1) and skip counting; within-rank also skip. Since those elements don't exist, they shouldn't contribute. In scatter I only iterate valid flat ids.

Actually for simplicity, let me have tile count kernel only cover the LAST partial tile for invalid, but to keep TILE uniform, mask properly.

Let me also reconsider possible overflow / types: counts up to n=32768, fits int32. seg_start up to total_padded ≤ ~163K fits int32. out_ids int32.

Let me now also handle the case where NT could be 1 and E could be large, etc. All fine.

Let me reconsider performance: kernel 1 within-tile rank O(TILE²)=1024² per CTA but that's ~ 1M shared ops. NT up to 32 CTAs, on 108 SMs → fine, single wave, ~ maybe 30-50us? The 1M shared ops per CTA with 1024 threads, each thread ~1024 iterations reading s_e (shared). Shared load latency ~ a few cycles, 1024 iterations → ~4096 cycles ~ 2-3us per CTA, 32 CTAs in parallel on 32 SMs → ~3us. 

Kernel 1 also does atomicAdd into s_cnt for each element — n total atomics across CTAs in shared, fast.

Alright, I'm fairly confident. Let me write the CUDA code via `load_inline`.

Let me also handle the `within_rank` storage — I'll store to a global `rank_buf` (int32, n). Actually, can I avoid storing and recompute in scatter? Scatter is a separate kernel; recomputing within-rank there means another O(TILE²). Better store. Or store rank per flat_id in kernel 1. Actually simplest: kernel 1 stores `within_rank[f]` into a global array. Let me name `tile_rank`.

Wait, I can simplify: instead of separate tile_rank array, note dest = seg_start[e] + tile_base[t,e] + within_rank. I need tile_base (from kernel 2, depends on tile_cnt) and within_rank (kernel 1). So I store within_rank (n ints) in kernel 1, then kernel 3 combines.

Alternatively, kernel 1 stores `dest_components`? No, needs kernel 2.

Let me write it.

Actually, let me reconsider: can I merge within_rank computation more cleanly? For thread i in tile (i = 0..TILE-1): 
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


_CPP_SRC = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> block_aligned_route_cuda(
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    int64_t num_experts,
    int64_t block_size);

std::vector<torch::Tensor> block_aligned_route(
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    int64_t num_experts,
    int64_t block_size) {
  return block_aligned_route_cuda(
      expert_ids, routing_weights, num_experts, block_size);
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void tile_rank_count_kernel(
    const int* __restrict__ expert_ids,
    int n,
    int num_experts,
    int* __restrict__ tile_counts,
    int* __restrict__ local_rank) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x) * 8 + warp;
  const int flat_id = tile * 32 + lane;
  const unsigned valid = __ballot_sync(0xffffffffu, flat_id < n);
  if (flat_id < n) {
    const int expert = expert_ids[flat_id];
    const unsigned peers = __match_any_sync(valid, expert);
    local_rank[flat_id] = __popc(peers & ((1u << lane) - 1u));
    if (lane == (__ffs(peers) - 1)) {
      tile_counts[tile * num_experts + expert] = __popc(peers);
    }
  }
}

__global__ void layout_kernel(
    const int* __restrict__ tile_counts,
    int num_tiles,
    int num_experts,
    int block_size,
    int* __restrict__ tile_base,
    int* __restrict__ counts,
    int* __restrict__ expert_start,
    int* __restrict__ num_post_pad) {
  __shared__ int padded_scan[512];
  const int tid = threadIdx.x;
  int count = 0;
  int padded = 0;
  if (tid < num_experts) {
    for (int tile = 0; tile < num_tiles; ++tile) {
      const int offset = tile * num_experts + tid;
      tile_base[offset] = count;
      count += tile_counts[offset];
    }
    counts[tid] = count;
    padded = ((count + block_size - 1) / block_size) * block_size;
  }
  padded_scan[tid] = padded;
  __syncthreads();
  #pragma unroll
  for (int offset = 1; offset < 512; offset <<= 1) {
    const int add = tid >= offset ? padded_scan[tid - offset] : 0;
    __syncthreads();
    padded_scan[tid] += add;
    __syncthreads();
  }
  if (tid < num_experts) {
    expert_start[tid] = padded_scan[tid] - padded;
  }
  if (tid == 511) {
    num_post_pad[0] = padded_scan[511];
  }
}

__global__ void initialize_kernel(
    int* __restrict__ sorted_ids,
    float* __restrict__ sorted_weights,
    int* __restrict__ block_expert,
    const int* __restrict__ counts,
    const int* __restrict__ expert_start,
    int n,
    int total,
    int num_experts,
    int block_size) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    sorted_ids[idx] = n;
    sorted_weights[idx] = 0.0f;
  }
  if (idx < num_experts) {
    const int nblocks = (counts[idx] + block_size - 1) / block_size;
    const int first_block = expert_start[idx] / block_size;
    for (int block = 0; block < nblocks; ++block) {
      block_expert[first_block + block] = idx;
    }
  }
}

__global__ void scatter_kernel(
    const int* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const int* __restrict__ local_rank,
    const int* __restrict__ tile_base,
    const int* __restrict__ expert_start,
    int* __restrict__ sorted_ids,
    float* __restrict__ sorted_weights,
    int n,
    int num_experts) {
  const int flat_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat_id < n) {
    const int expert = expert_ids[flat_id];
    const int tile = flat_id >> 5;
    const int dst = expert_start[expert]
                  + tile_base[tile * num_experts + expert]
                  + local_rank[flat_id];
    sorted_ids[dst] = flat_id;
    sorted_weights[dst] = routing_weights[flat_id];
  }
}

}  // namespace

std::vector<torch::Tensor> block_aligned_route_cuda(
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    int64_t num_experts_i64,
    int64_t block_size_i64) {
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be CUDA");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be CUDA");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32,
              "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32,
              "routing_weights must be float32");
  TORCH_CHECK(expert_ids.is_contiguous() && routing_weights.is_contiguous(),
              "inputs must be contiguous");
  const int n = static_cast<int>(expert_ids.numel());
  const int num_experts = static_cast<int>(num_experts_i64);
  const int block_size = static_cast<int>(block_size_i64);
  TORCH_CHECK(num_experts > 0 && num_experts <= 512,
              "num_experts must be in [1, 512]");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int num_tiles = (n + 31) / 32;
  auto int_opts = expert_ids.options().dtype(torch::kInt32);
  auto float_opts = routing_weights.options().dtype(torch::kFloat32);
  auto tile_counts = torch::empty({num_tiles * num_experts}, int_opts);
  auto tile_base = torch::empty({num_tiles * num_experts}, int_opts);
  auto local_rank = torch::empty({n}, int_opts);
  auto counts = torch::empty({num_experts}, int_opts);
  auto expert_start = torch::empty({num_experts}, int_opts);
  auto num_post_pad = torch::empty({1}, int_opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(expert_ids.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      tile_counts.data_ptr<int>(), 0,
      static_cast<size_t>(num_tiles) * num_experts * sizeof(int), stream));
  tile_rank_count_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      expert_ids.data_ptr<int>(), n, num_experts,
      tile_counts.data_ptr<int>(), local_rank.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  layout_kernel<<<1, 512, 0, stream>>>(
      tile_counts.data_ptr<int>(), num_tiles, num_experts, block_size,
      tile_base.data_ptr<int>(), counts.data_ptr<int>(),
      expert_start.data_ptr<int>(), num_post_pad.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int total = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &total, num_post_pad.data_ptr<int>(), sizeof(int),
      cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto sorted_ids = torch::empty({total}, int_opts);
  auto sorted_weights = torch::empty({total}, float_opts);
  auto block_expert = torch::empty({total / block_size}, int_opts);

  const int init_n = total > num_experts ? total : num_experts;
  initialize_kernel<<<(init_n + 255) / 256, 256, 0, stream>>>(
      sorted_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(),
      block_expert.data_ptr<int>(), counts.data_ptr<int>(),
      expert_start.data_ptr<int>(), n, total, num_experts, block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  scatter_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      expert_ids.data_ptr<int>(), routing_weights.data_ptr<float>(),
      local_rank.data_ptr<int>(), tile_base.data_ptr<int>(),
      expert_start.data_ptr<int>(), sorted_ids.data_ptr<int>(),
      sorted_weights.data_ptr<float>(), n, num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted_ids, sorted_weights, block_expert, num_post_pad};
}
"""


@triton.jit
def _rank_count_kernel(expert_ids, tile_counts, local_rank,
                       N: tl.constexpr, NT: tl.constexpr,
                       TILE: tl.constexpr):
    pid = tl.program_id(0)
    expert = pid // NT
    tile = pid - expert * NT
    offsets = tile * TILE + tl.arange(0, TILE)
    valid = offsets < N
    values = tl.load(expert_ids + offsets, mask=valid, other=-1)
    match = valid & (values == expert)
    flags = match.to(tl.int32)
    rank = tl.cumsum(flags, axis=0) - 1
    tl.store(local_rank + offsets, rank, mask=match)
    tl.store(tile_counts + expert * NT + tile, tl.sum(flags, axis=0))


@triton.jit
def _tile_prefix_kernel(tile_counts, tile_base, counts,
                        NT: tl.constexpr, BLOCK_T: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    expert = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_T)
    mask = tiles < NT
    values = tl.load(tile_counts + expert * NT + tiles, mask=mask, other=0)
    bases = tl.cumsum(values, axis=0) - values
    tl.store(tile_base + expert * NT + tiles, bases, mask=mask)
    tl.store(counts + expert, tl.sum(values, axis=0))


@triton.jit
def _rank_prefix_kernel(expert_ids, local_rank, counts,
                        N: tl.constexpr, NT: tl.constexpr,
                        TILE: tl.constexpr):
    expert = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    running = 0
    for tile in range(0, NT):
        offsets = tile * TILE + lanes
        valid = offsets < N
        values = tl.load(expert_ids + offsets, mask=valid, other=-1)
        match = valid & (values == expert)
        flags = match.to(tl.int32)
        rank = tl.cumsum(flags, axis=0) - 1
        tl.store(local_rank + offsets, running + rank, mask=match)
        running += tl.sum(flags, axis=0)
    tl.store(counts + expert, running)


@triton.jit
def _expert_prefix_kernel(counts, expert_start, num_post_pad,
                          E: tl.constexpr, BLOCK_E: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    experts = tl.arange(0, BLOCK_E)
    mask = experts < E
    count = tl.load(counts + experts, mask=mask, other=0)
    padded = ((count + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    start = tl.cumsum(padded, axis=0) - padded
    tl.store(expert_start + experts, start, mask=mask)
    tl.store(num_post_pad, tl.sum(padded, axis=0))


@triton.jit
def _initialize_kernel(sorted_ids, sorted_weights, block_expert,
                       counts, expert_start,
                       N: tl.constexpr, TOTAL: tl.constexpr,
                       E: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                       BLOCK_OUT: tl.constexpr, MAX_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = offsets < TOTAL
    tl.store(sorted_ids + offsets, N, mask=out_mask)
    tl.store(sorted_weights + offsets, 0.0, mask=out_mask)

    expert_mask = pid < E
    count = tl.load(counts + pid, mask=expert_mask, other=0)
    start = tl.load(expert_start + pid, mask=expert_mask, other=0)
    blocks = tl.arange(0, MAX_BLOCKS)
    nblocks = (count + BLOCK_SIZE - 1) // BLOCK_SIZE
    tl.store(block_expert + start // BLOCK_SIZE + blocks, pid,
             mask=expert_mask & (blocks < nblocks))


@triton.jit
def _scatter_kernel(expert_ids, routing_weights, local_rank,
                    counts, expert_start,
                    sorted_ids, sorted_weights, block_expert,
                    N: tl.constexpr,
                    E: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                    TILE: tl.constexpr, BLOCK_N: tl.constexpr,
                    MAX_PAD: tl.constexpr,
                    MAX_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N
    expert = tl.load(expert_ids + offsets, mask=mask, other=0)
    rank = tl.load(local_rank + offsets, mask=mask, other=0)
    start = tl.load(expert_start + expert, mask=mask, other=0)
    dst = start + rank
    weight = tl.load(routing_weights + offsets, mask=mask, other=0.0)
    tl.store(sorted_ids + dst, offsets, mask=mask)
    tl.store(sorted_weights + dst, weight, mask=mask)

    expert_mask = pid < E
    count = tl.load(counts + pid, mask=expert_mask, other=0)
    seg_start = tl.load(expert_start + pid, mask=expert_mask, other=0)
    padded = ((count + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    pads = tl.arange(0, MAX_PAD)
    pad_mask = expert_mask & (pads < padded - count)
    pad_dst = seg_start + count + pads
    tl.store(sorted_ids + pad_dst, N, mask=pad_mask)
    tl.store(sorted_weights + pad_dst, 0.0, mask=pad_mask)

    blocks = tl.arange(0, MAX_BLOCKS)
    nblocks = padded // BLOCK_SIZE
    tl.store(block_expert + seg_start // BLOCK_SIZE + blocks, pid,
             mask=expert_mask & (blocks < nblocks))


class ModelNew(nn.Module):
    def forward(self, expert_ids, routing_weights, num_experts, block_size):
        n = expert_ids.numel()
        e = int(num_experts)
        block_size = int(block_size)
        tile = 1024
        nt = triton.cdiv(n, tile)
        device = expert_ids.device

        workspace = torch.empty(
            (n + 2 * e + 1,), dtype=torch.int32, device=device
        )
        cursor = 0
        local_rank = workspace[cursor:cursor + n]
        cursor += n
        counts = workspace[cursor:cursor + e]
        cursor += e
        expert_start = workspace[cursor:cursor + e]
        cursor += e
        num_post_pad = workspace[cursor:cursor + 1]

        _rank_prefix_kernel[(e,)](
            expert_ids, local_rank, counts,
            N=n, NT=nt, TILE=tile, num_warps=4,
        )
        block_e = triton.next_power_of_2(e)
        _expert_prefix_kernel[(1,)](
            counts, expert_start, num_post_pad,
            E=e, BLOCK_E=block_e, BLOCK_SIZE=block_size,
            num_warps=4,
        )

        total = int(num_post_pad.item())
        total_blocks = total // block_size
        output_storage = torch.empty(
            (2 * total + total_blocks,), dtype=torch.int32, device=device
        )
        sorted_ids = output_storage[:total]
        sorted_weights = output_storage[total:2 * total].view(torch.float32)
        block_expert = output_storage[2 * total:2 * total + total_blocks]

        max_blocks = triton.next_power_of_2(triton.cdiv(n, block_size))
        scatter_block = 256
        scatter_grid = max(triton.cdiv(n, scatter_block), e)
        _scatter_kernel[(scatter_grid,)](
            expert_ids, routing_weights, local_rank, counts,
            expert_start, sorted_ids, sorted_weights, block_expert,
            N=n, E=e, BLOCK_SIZE=block_size, TILE=tile,
            BLOCK_N=scatter_block,
            MAX_PAD=triton.next_power_of_2(block_size),
            MAX_BLOCKS=max_blocks, num_warps=8,
        )
        return sorted_ids, sorted_weights, block_expert, num_post_pad
