package Problems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Vector;

import Behaviors.Behavior;
import Behaviors.RandomBehavior;

public class Maze extends Problem {
	public static final int FREE= 0;
	public static final int WALL = 1;
	
	public int curStep=0;
	public ArrayList<Action> possibleActions=new ArrayList<Action>();
	
	public class Position extends State {
		public int x;
		public int y;

		public Position(int x, int y) {
			this.x=x;
			this.y=y;
		}
		
		@Override
		public boolean equals(State s) {
			if (s instanceof Position) {
				Position p = (Position) s;
				return p.x==x && p.y==y;
			} else {
				return false;
			}
		}
	}
	
	public class Move extends Action {
		public int dx;
		public int dy;
		
		public Move(int dx, int dy) {
			this.dx=dx;
			this.dy=dy;
		}

		@Override
		public boolean equals(Action a) {
			if (a instanceof Move) {
				Move m = (Move) a;
				return m.dx==dx && m.dy==dy;
			} else {
				return false;
			}
		}

	}
	
	public int map[][];
	Position in;
	Position out;
	Position curPos;
	
	int width;
	int height;
	
	// build a default maze, from sutton book
	public Maze(Behavior b) {
		super(b);
		
		out = new Position(9, 1);
		in =new Position(1, 3);
		curPos=new Position(1, 3);
		
		width=11;
		height=8;
		map = new int[width][height];
		for (int x=0; x<width; x++) {
			for (int y=0; y<height; y++) {
				map[x][y]=FREE;
			}
		}
		for (int x=0; x<width; x++) {
			map[x][0]=WALL;
			map[x][7]=WALL;
		}
		for (int y=1; y<height; y++) {
			map[0][y]=WALL;
			map[10][y]=WALL;
		}
		map[3][2]=WALL;
		map[3][3]=WALL;
		map[3][4]=WALL;
		map[6][5]=WALL;
		map[8][1]=WALL;
		map[8][2]=WALL;
		map[8][3]=WALL;
		
		possibleActions.add(new Move(-1,  0)); // LEFT
		possibleActions.add(new Move( 1,  0)); // RIGHT
		possibleActions.add(new Move( 0, -1)); // TOP
		possibleActions.add(new Move( 0,  1)); // BOTTOM
	}

	
	
	@Override
	public double run(int maxSteps) {
		while (curStep<maxSteps && !isSolved()) {
			stepForward();
		}
		return (curPos.equals(out)?1.0:0.0);
	}
	
	@Override
	public void stepForward() {
		Move a = (Move)behavior.chooseAction(curPos, possibleActions);
		if (map[curPos.x+a.dx][curPos.y+a.dy] != WALL) {
			curPos.x+=a.dx;
			curPos.y+=a.dy;
		}
		curStep++;
	}

	@Override
	public boolean isSolved() {
		return curPos.equals(out);
	}
	
	@Override
	public double getReward(State s) {
		if (s instanceof Position) {
			Position p = (Position) s;
			if (p.x==out.x && p.y==out.y) {
				return 1.0;
			} else {
				return 0.0;
			}
		} else {
			return 0.0;
		}
	}
	
	@Override
	public String displayCurrentState() {
		String res=new String();
		for (int y=0; y<height; y++) {
			for (int x=0; x<width; x++) {
				if (x==curPos.x && y==curPos.y) res+='@';
				else if (x==in.x && y==in.y) res+='I';
				else if (x==out.x && y==out.y) res+='O';
				else if (map[x][y]==FREE) res+=' ';
				else if (map[x][y]==WALL) res+='#';
			}
			res+='\n';
		}
		return res;
	}
	
	@Override
	public int getNStates() {
		return width*height;
	}
	
	@Override
	public int getNActions() {
		return possibleActions.size();
	}

	public static void main(String[] args) {
		RandomBehavior behavior = new RandomBehavior(System.currentTimeMillis());
		Maze maze=new Maze(behavior);
		System.out.println(maze.displayCurrentState());
		while (maze.curStep<1000 && !maze.isSolved()) {
			maze.stepForward();
			System.out.println("============\t"+maze.curStep+"\t============");
			System.out.println(maze.displayCurrentState());
		}
	}
	
}
