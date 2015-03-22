package learning;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;
import Behaviors.Behavior;
import Problems.Action;
import Problems.Problem;
import Problems.State;

public class SARSA extends Behavior {
	Random rng;
	public DoubleMatrix2D e;
	public DoubleMatrix2D q;
	
	float epsilon = 0;
	int policy = -1;
	public double lambda;
	public double gamma;
	public double alpha;
	
	public int verbosity = 0;// 0 is silent, 1 is episode, 100 is display all states

	public SARSA(int nStates, int nActions, long rngSeed) {
		super(nStates, nActions);
		
		rng=new Random(rngSeed);
		this.e = new DenseDoubleMatrix2D( nStates, nActions );
		this.q = new DenseDoubleMatrix2D( nStates, nActions );
	}	
	
	public Action chooseAction(State s, ArrayList<Action> possibleActions) {
		double[] actionQs = new double[nActions];
		int[] greedyAs = new int[nActions];
		double greedyVal = Double.NEGATIVE_INFINITY;
		int greedyCount = 0;
		for( int a = 0; a < nActions; a++ ) {
			actionQs[a] = q.get( s.index(), a );
			if( actionQs[a] > greedyVal ) {
				greedyVal = actionQs[a];
				Arrays.fill( greedyAs, 0 );
				greedyCount = 1;
				greedyAs[a] = 1;
			} else if( actionQs[a] == greedyVal ) {
				greedyAs[a] = 1;
				greedyCount++;
			}
		}
		//System.out.println( "! " + greedyCount + " " + greedyVal );
		int greedyRand = rng.nextInt( greedyCount );
		int greedyA = 0;
		for( int k = 0; k < nActions; k++ ) {
			if( greedyRand == 0 && greedyAs[k] == 1 ) {
				greedyA = k;
				break;
			} else if( greedyRand > 0 && greedyAs[k] == 1 ) {
				greedyRand--;
			} 
		}
		
		if( policy == 0 ) { // greedy
			return possibleActions.get(greedyA);
		} else if( policy == 1 ) { // epsilon greedy
			
			if( rng.nextDouble() < epsilon )// Random action 
				return possibleActions.get( rng.nextInt( nActions ) );
			else// Greedy action
				return possibleActions.get(greedyA);
		} else {
			return null;// break stuff if things dont work as they should
		}
		
	}
	
	DoubleDoubleFunction plus = new DoubleDoubleFunction() {
	    public double apply(double a, double b) { return a+b; }
	};
	DoubleDoubleFunction elmul = new DoubleDoubleFunction() {
	    public double apply(double a, double b) { return a*b; }
	};
	
	public void training( Problem p, String[] policyArguments, int numEpisodes, int maxSteps ) {
		// Zero out eligibility trace and q
		e.assign( 0 );
		q.assign( 0 );
		
		// Yes, String policies are gross, but it is only needed for non GRRL policies		
		if( policyArguments[0] == "greedy" ) {
			policy = 0;
		} else if( policyArguments[0] == "epsilonGreedy" ) {
			epsilon = Float.parseFloat( policyArguments[1] );
			policy = 1;
		}		
		
		double runningScore = 0;
		
		DenseDoubleMatrix2D escale = new DenseDoubleMatrix2D( nStates, nActions );
		//escale.assign( gamma * lambda );
		
		// Now start the training
		for( int episode = 0; episode < numEpisodes; episode++ ) {
			p.reset();// Initialize problem

			e.assign( 0 );// Eligibility traces are reset for each episode
			//System.out.println( "e sum " + e.zSum() );
			
			while (p.curStep<1000 && !p.isSolved()) {												
				State curState = p.getCurrentState();
				Action curAction = this.chooseAction( curState, p.possibleActions );				
				
				p.stepForward(curAction);
				
				State nextState = p.getCurrentState();
				
				if( !curState.equals( nextState ) ) {
				
				
				double reward = p.getReward( nextState );
								
				Action nextAction = this.chooseAction( nextState, p.possibleActions );
				
				double curQ = q.get( curState.index(), curAction.index );
				double nextQ = q.get( nextState.index(), nextAction.index );
				
				double delta = reward + gamma * nextQ - curQ;
				
				if( verbosity > 10 )
					System.out.println( "Delta = " + delta + " curQ " + curQ + " nextQ " + nextQ + " reward " + reward);
				//double updatedQ = curQ + alpha * ( delta );
				
				// Increment state/action by 1
				e.set( curState.index(), curAction.index, 
						e.get( curState.index(), curAction.index ) + 1 );
				
				// Trace * alpha * delta matrix
				escale = (DenseDoubleMatrix2D) e.copy();
				escale.assign( DoubleFunctions.mult( alpha * delta ) );
				
				// Update Qs
				q.assign( escale, DoubleFunctions.plus );
				
				// Update Es				
				// Scale traces
				e.assign( DoubleFunctions.mult( gamma * lambda ) );				
				
				if( verbosity > 10 ) {
					System.out.println("============\t"+p.curStep+"\t============");
					System.out.println(p.displayCurrentState());
				}
				}
			}
						
			runningScore = p.curStep + 0.95 * runningScore; 
			
			if( verbosity > 0 ) {
				System.out.println( "Finished episode " + episode + " in " + p.curStep + " steps. Running score " + runningScore );
			}
			
		}
	}
	
	public void saveMatrixtoCSV( DoubleMatrix2D matrix, String filename ) {
		try {
			FileWriter fWriter = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fWriter);
			
			int rows = matrix.rows();
            int cols = matrix.columns();
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < cols; j++){
                        writer.write(matrix.get(i, j) + "\t");
                }
                writer.write("\n");
            }
            writer.close();
			
		}catch(IOException ex){
            ex.printStackTrace();
		}
	}
	
}
