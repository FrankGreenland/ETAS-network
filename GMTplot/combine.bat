# 这是将多个地区的图拼接而成的
gmt begin map2 jpg
  gmt subplot begin 2x2 -Fs30c/30c -A+jTL -M0.2c/1c

    gmt subplot set 0
    # 山西断陷带
    gmt basemap -R110.0/115.0/34.0/41.0 -JM? -Ba1f1g1 -BWSen
    gmt plot shanxi_vector.dat -S=0.5c -W0.3p -Ggrey -JM? -R110.0/115.0/34.0/41.0 -t70

    gmt plot CN-faults.gmt -W0.5 -Z -Csaddlebrown

    gmt makecpt -Cplasma -T0/0.15/0.001
    gmt plot shanxi_points.dat -Sc -C -l"Vertex"

    gmt makecpt -Cplasma -T0/0.15/0.001 -H > Icpt.cpt
    # gmt colorbar -CIcpt.cpt -Dn0.5/0.25+jCM+w15c/0.25c+h+e+n -B+l"Master CPT"
    gmt colorbar -CIcpt.cpt -Dn1.1/0.5+jCM+w20c/0.35c+mc -B



    gmt subplot set 1  
    # -Cx+0.1c
    # INGV 意大利
    gmt basemap -R5/20/35/48 -JM? -Ba1f1g1 -BWSen
    gmt plot INGV_vector.dat -S=0.5c -W0.3p -Ggrey -JM? -R5/20/35/48 -t80

    gmt plot gem_active_faults.gmt -W0.7 -Z -Csaddlebrown

    gmt makecpt -Cplasma -T0/0.05/0.0001
    gmt plot INGV_points.dat -Sc -C -l"Vertex"

    gmt makecpt -Cplasma -T0/0.05/0.0001 -H > Icpt.cpt
    # gmt colorbar -CIcpt.cpt -Dn0.5/0.25+jCM+w15c/0.25c+h+e+n -B+l"Master CPT"
    gmt colorbar -CIcpt.cpt -Dn1.1/0.5+jCM+w20c/0.35c+mc -B



    gmt subplot set 2
    # chuandian
    gmt basemap -R98/106/22/33 -JM? -Ba1f1g1 -BWSen
    gmt plot chuandian_vector.dat -S=0.5c -W0.3p -Ggrey -JM? -R98/106/22/33 -t70

    gmt plot CN-faults.gmt -W0.5 -Z -Csaddlebrown

    gmt makecpt -Cplasma -T0/0.083/0.001
    gmt plot chuandian_points.dat -Sc -C -l"Vertex"

    gmt makecpt -Cplasma -T0/0.083/0.001 -H > Icpt.cpt
    gmt colorbar -CIcpt.cpt -Dn1.1/0.5+jCM+w20c/0.35c+mc -B





    gmt subplot set 3
    # 南加州
    gmt basemap -R-122/-114/32/37 -JM? -Ba1f1g1 -BWSen
    gmt plot southern_california_vector.dat -S=0.5c -W0.3p -Ggrey -JM? -R-122/-114/32/37 -t80
    gmt plot gem_active_faults.gmt -W0.7 -Z -Csaddlebrown

    gmt makecpt -Cplasma -T0/0.084/0.0001
    gmt plot southern_california_points.dat -Sc -C -l"Vertex"
    gmt makecpt -Cplasma -T0/0.084/0.0001 -H > Icpt.cpt
    gmt colorbar -CIcpt.cpt -Dn1.1/0.5+jCM+w20c/0.35c+mc -B

  gmt subplot end
gmt end show