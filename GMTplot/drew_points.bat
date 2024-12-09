gmt begin seisnetwork_shanxi jpg
    # 加个底图
    gmt basemap -JM15c -R110.0/115.0/34.0/41.0 -Ba1f1 -BWSen -Bg1

    # 地势与地势的colorbar
    # gmt grdimage @earth_relief_15s -JM105/35/10c -R110.0/115.0/34.0/41.0 -Ba1f1 -BWSen
    # gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"




    # 加上连接线 （箭头） velo 或者plot -Sv
    # gmt plot vector.dat -S=0.5c+eA -W0.25p -Ggrey -JM15c -R110.0/115.0/34.0/41.0 -t70
    gmt plot vector.dat -S=0.5c -W0.15p -Ggrey -JM15c -R110.0/115.0/34.0/41.0 -t90

    # 断层线
    gmt plot CN-faults.gmt -W0.5 -Z -Csaddlebrown -l"fault"

    # 画上地震点点， 需要确定大小和颜色
    # 这个是固定大小与颜色 → gmt plot shanxi_position -Sc0.075 -Glightblue -l"earthquake"
    

    gmt makecpt -Cplasma -T0/1/0.001
    # gmt plot shanxi_point_size.dat -Sc0.5c -C -l"Vertex"
    # -Cmag  -Iz
    gmt plot -t50 points.dat -Sc -C -l"Vertex"

    gmt makecpt -Cplasma -T0/1/0.001 -H  > Icpt.cpt
    # gmt colorbar -CIcpt.cpt -Dn0.5/0.25+jCM+w15c/0.25c+h+e+n -B+l"Master CPT"
    gmt colorbar -CIcpt.cpt -Dn1.1/0.5+jCM+w20c/0.35c+v+e+mc -B+l"Property"


    # 控制legend的位置
    # gmt legend -DjBR+o-5c/3c -F+gwhite -S1.3
gmt end show