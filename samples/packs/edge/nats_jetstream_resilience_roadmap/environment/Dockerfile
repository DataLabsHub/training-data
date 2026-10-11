FROM golang:1.23.7-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*

COPY dependency-cache-go.mod /tmp/roadmap-deps/go.mod
COPY dependency-cache-go.sum /tmp/roadmap-deps/go.sum
RUN cd /tmp/roadmap-deps \
    && go mod download \
    && rm -rf /tmp/roadmap-deps

COPY repo/go.mod /tmp/baseline-deps/go.mod
COPY repo/go.sum /tmp/baseline-deps/go.sum
RUN cd /tmp/baseline-deps \
    && go mod download \
    && rm -rf /tmp/baseline-deps

WORKDIR /app
COPY repo/ /app/
RUN go mod download \
    && go test ./... -run '^$'

ENV GOPROXY=off
ENV GOSUMDB=off
CMD ["bash"]
