gmt begin seismic_shanxi jpg
    gmt grdimage @earth_relief_15s -JM105/35/10c -R110.0/115.0/34.0/41.0 -Ba1f1 -BWSen
	gmt grdcut -R110.0/115.0/34.0/41.0@earth_relief_02m -Gtrack.grd
	gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"
	gmt plot shanxi_position -Sc0.075 -Glightblue -l"earthquake"
	gmt plot CN-faults.gmt -W0.5 -Z -Cred -l"fault"
	gmt legend -DjBR+o-5c/3c -F+gwhite -S1.3
gmt end show