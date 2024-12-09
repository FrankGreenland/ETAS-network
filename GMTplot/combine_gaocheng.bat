# 这是将多个地区的图拼接而成的
# 只绘制高程和地震点位
gmt begin gaocheng_map jpg
  gmt subplot begin 2x2 -Fs30c/30c -A+jTL+o-0.5c/1c -M1c/1c

    gmt subplot set 0
    # 山西断陷带
    # gmt makecpt -Ccopper -T0/3000
    # gmt makecpt -Cfes -T0/3000
    gmt makecpt -Coleron -T0/3000
    gmt grdimage @earth_relief_15s -JM? -R110.0/115.0/34.0/41.0 -Ba1f1 -BWSen
    # gmt basemap -JM? -R110.0/115.0/34.0/41.0 -Ba1f1 -BWSen
    # gmt coast -JM? -R110.0/115.0/34.0/41.0 -Ggray95
    gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"
    gmt plot position/shanxi_position.dat -Sc -GKHAKI
    gmt plot CN-faults.gmt -W2.0 -Z -Cred


    gmt subplot set 1  
    # -Cx+0.1c
    # INGV 意大利
    # gmt makecpt -CgrayC -T-4000/4500
    # gmt makecpt -Ccopper -T-4000/4500
    gmt makecpt -Coleron -T-4000/3000
    gmt grdimage @earth_relief_15s -JM? -R5/20/35/48 -Ba1f1 -BWSen
    gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"
    gmt plot position/INGV_position.dat -Sc -GKHAKI
    gmt plot gem_active_faults.gmt -W2.0 -Z -Cred



    gmt subplot set 2
    # chuandian
    # gmt makecpt -Ccopper -T0/10000
    gmt makecpt -Coleron -T0/6800
    gmt grdimage @earth_relief_15s -JM? -R98/106/22/33 -Ba1f1 -BWSen
    gmt plot position/chuandian_position.dat -Sc -GKHAKI
    gmt plot CN-faults.gmt -W2.0 -Z -Cred
    gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"


    gmt subplot set 3
    # 南加州
    # gmt makecpt -Ccopper -T-3000/4000
    gmt makecpt -Coleron -T-3000/4000
    gmt grdimage @earth_relief_15s -JM? -R-122/-114/32/37 -Ba1f1 -BWSen
    gmt plot position/sc_position.dat -Sc -GKHAKI
    gmt plot gem_active_faults.gmt -W2.0 -Z -Cred
    gmt colorbar -DJMR+ml -Bxa500f -By+l"(m)"



  gmt subplot end
gmt end show