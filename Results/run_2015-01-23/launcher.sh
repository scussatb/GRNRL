#!/bin/sh

CROSSRATE=0.2
MUTRATE=0.8
MAXNUMGEN=500
SPETHRE=0.15
SAVEPOP=false
EXPNAME=NMSarsa
DUPINIT=19
REPMET=1
SSAM=2
XOVER=4
NCPU=4
PROBLEMS=$@

for RS in 1212 1111 1010 999
do
	echo "Launching $PROBLEMS with random seed $RS"
	nice java -Xmx512m -jar evolver.jar great true nCPU $NCPU maxNumGen $MAXNUMGEN xover $XOVER speciationThreshold $SPETHRE saveAllPopulation $SAVEPOP randomSeed $RS experienceName $EXPNAME crossoverRate $CROSSRATE mutationRate $MUTRATE duplicateInit $DUPINIT representativeMethod $REPMET speciesSizeAdjustingMethod $SSAM problems $PROBLEMS endProblems > out_${PROBLEMS}_${RS}.log
done

for job in `jobs -p`
do 
	echo "Waiting job $job"
	wait $job
done


 
