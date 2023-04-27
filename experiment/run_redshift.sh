for i in {101..120..1} #Repeats for BART in different seeds
do
    for j in 1 2 5 10 20 #number of trees in BART
    do
    	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.001smallp7_99.csv redshift_y_train_0.001smallp7_99.csv redshift_x_test_0.001smallp7_99.csv redshift_y_test_0.001smallp7_99.csv redshift0001"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.005smallp7_99.csv redshift_y_train_0.005smallp7_99.csv redshift_x_test_0.005smallp7_99.csv redshift_y_test_0.005smallp7_99.csv redshift0005"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.010smallp7_99.csv redshift_y_train_0.010smallp7_99.csv redshift_x_test_0.010smallp7_99.csv redshift_y_test_0.010smallp7_99.csv redshift0010"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.050smallp7_99.csv redshift_y_train_0.050smallp7_99.csv redshift_x_test_0.050smallp7_99.csv redshift_y_test_0.050smallp7_99.csv redshift0050"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.100smallp7_99.csv redshift_y_train_0.100smallp7_99.csv redshift_x_test_0.100smallp7_99.csv redshift_y_test_0.100smallp7_99.csv redshift0100"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.150smallp7_99.csv redshift_y_train_0.150smallp7_99.csv redshift_x_test_0.150smallp7_99.csv redshift_y_test_0.150smallp7_99.csv redshift0150"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.200smallp7_99.csv redshift_y_train_0.200smallp7_99.csv redshift_x_test_0.200smallp7_99.csv redshift_y_test_0.200smallp7_99.csv redshift0200"_"$i"_"$j
	Rscript run_bigBatchBART_redshift.R $i $j redshift_x_train_0.250smallp7_99.csv redshift_y_train_0.250smallp7_99.csv redshift_x_test_0.250smallp7_99.csv redshift_y_test_0.250smallp7_99.csv redshift0250"_"$i"_"$j
    done
done

