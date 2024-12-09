# 这个脚本是画box分盒的示意图的
# 选择的是INGV
gmt begin multipolygons_INGV pdf
    gmt basemap -R5/20/35/48 -JM10c -Ba2f2 -BWSen
    gmt plot boxes_full.dat  -L -Gnavajowhite2
    gmt plot boxes_empty.dat -W0.01p,black -L
    gmt plot gem_active_faults.gmt -W0.7 -Z -Csaddlebrown
gmt end show