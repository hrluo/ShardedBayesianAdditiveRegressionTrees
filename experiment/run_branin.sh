for i in {101..120..1} #Repeats for BART in different seeds
do
    for j in 1 2 5 #number of trees in BART
    do
    	for k in 99 #random seed of training and testing sets.
    	do
    		for l in 2 15 #dimension of X
    		do
			Rscript run_bigBatchBART.R $i $j x100p$l"_"$k.csv y100p$l"_"$k.csv xp100p$l"_"$k.csv yp100p$l"_"$k.csv branin100p$l"_"$k
			Rscript run_bigBatchBART.R $i $j x1000p$l"_"$k.csv y1000p$l"_"$k.csv xp1000p$l"_"$k.csv yp1000p$l"_"$k.csv branin1000p$l"_"$k
			Rscript run_bigBatchBART.R $i $j x10000p$l"_"$k.csv y10000p$l"_"$k.csv xp10000p$l"_"$k.csv yp10000p$l"_"$k.csv branin10000p$l"_"$k
		done
        done
    done
done

