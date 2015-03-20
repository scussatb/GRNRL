package Behaviors;

import java.util.ArrayList;
import java.util.Random;

import Problems.Action;
import Problems.State;

public class RandomBehavior extends Behavior {
	Random rng;
	
	public RandomBehavior(long rngSeed) {
		super(0,0);
		rng=new Random(rngSeed);
	}

	@Override
	public Action chooseAction(State s, ArrayList<Action> possibleActions) {
		return possibleActions.get(rng.nextInt(possibleActions.size()));
	}

}
