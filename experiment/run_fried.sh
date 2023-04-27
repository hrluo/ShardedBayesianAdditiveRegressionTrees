for i in {101..120..1} #Repeats for BART in different seeds
do
    for j in 1 2 5 #number of trees in BART
    do
    	for k in 99 #random seed of training and testing sets.
    	do
    		for l in 5 20 100 1000 #dimension of X
    		do
			Rscript run_bigBatchBART.R $i $j fried_x5000p$l"_"$k.csv fried_y5000p$l"_"$k.csv fried_xp5000p$l"_"$k.csv fried_yp5000p$l"_"$k.csv fried_5000p$l"_"$k
		done
        done
    done
done

