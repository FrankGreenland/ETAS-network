# 这个脚本是画box分盒的示意图的
# 选择的是southern_california
gmt begin multipolygons_SC pdf
    gmt basemap -R-122/-114/32/37 -JM20c -Ba1f1 -BWSen
    gmt plot boxes_full.dat  -L -Gnavajowhite2
    gmt plot boxes_empty.dat -W0.2p,black -L
    gmt plot gem_active_faults.gmt -W1.4 -Z -Csaddlebrown
gmt end show