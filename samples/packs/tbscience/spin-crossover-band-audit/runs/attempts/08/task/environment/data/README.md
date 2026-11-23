# Variable-temperature vibrational spectra of an Fe(II) spin-crossover complex

Raw data supplied together with the two published assignment sheets.  The four
sample temperatures are 300, 200, 150 and 100 K.

## spectra/

Sixteen two-column CSV files, one per technique and temperature:

* `spectra/raman_532nm_<T>K.csv` - Raman, 532 nm excitation
* `spectra/raman_785nm_<T>K.csv` - Raman, 785 nm excitation
* `spectra/raman_1064nm_<T>K.csv` - Raman, 1064 nm excitation
* `spectra/ir_<T>K.csv` - infrared absorbance

`<T>` is one of `300`, `200`, `150` or `100`.  Every file carries the header

    wavenumber_cm1,intensity

followed by 2051 rows on a uniform 1 cm-1 grid from 150 to 2200 cm-1.  The
intensity scale is in arbitrary units; it is comparable within a file but not
between files, because each technique carries its own response factor and the
1064 nm data are noticeably noisier than the rest.

## bands.csv

The consolidated list of the 18 bands the audit must cover:

    band_id,center_cm1

`center_cm1` is the tabulated band centre; reproduce it as given.

## sheets/

The two published assignment sheets, `sheets/sheet_A.csv` and
`sheets/sheet_B.csv`, each list all 18 bands with the columns

    band_id,center_cm1,mode_label,marker

`mode_label` is the vibrational assignment that sheet proposes.  `marker`
names the spin state the sheet believes the band marks (`LS` or `HS`); an
empty field means the sheet does not treat that band as a marker.  The two
sheets disagree about several assignments, and their marker lists are disjoint
(no band is a marker on both sheets).

## susceptibility.csv

    T_K,chi_M_T_cm3Kmol,gamma_HS

`chi_M_T_cm3Kmol` is the measured molar magnetic susceptibility-temperature
product and `gamma_HS` the high-spin fraction derived from it, at each of the
four temperatures.
