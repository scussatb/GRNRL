#!/bin/sh

CROSSRATE=0.2
MUTRATE=0.8
MAXNUMGEN=150
SPETHRE=0.15
SAVEPOP=false
EXPNAME=NMSarsa
DUPINIT=0
REPMET=1
SSAM=2
XOVER=4
PROBLEMS=$@

for RS in 111 222 333 444 555 666 777 888 999 1010 1111 1212 1313 1414 1515 1616
do
	echo "Launching $PROBLEMS with random seed $RS"
	nice java -Xmx512m -jar great.jar great true maxNumGen $MAXNUMGEN xover $XOVER speciationThreshold $SPETHRE saveAllPopulation $SAVEPOP randomSeed $RS experienceName $EXPNAME crossoverRate $CROSSRATE mutationRate $MUTRATE duplicateInit $DUPINIT representativeMethod $REPMET speciesSizeAdjustingMethod $SSAM problems $PROBLEMS endProblems > out_${PROBLEMS}_${RS}.log
done


 
