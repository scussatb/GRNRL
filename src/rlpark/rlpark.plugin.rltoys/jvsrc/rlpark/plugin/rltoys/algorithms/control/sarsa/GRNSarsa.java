package rlpark.plugin.rltoys.algorithms.control.sarsa;

import java.util.ArrayList;

import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import grn.GRNModel;

public class GRNSarsa extends Sarsa {
	private static final long serialVersionUID = 4783867207342502849L;
	public static double minDelta = -100.0;
	public static double maxDelta = +100.0;
	public double alphaNorm=1.0;
	public int step=0;
	public int stepMax=100;
	public int episode=0;
	public double sumAvgE=0.0;
	public ArrayList<Double> avgsE=new ArrayList<Double>();
	public double sumD=0.0;
	public ArrayList<Double> sumsD=new ArrayList<Double>();
	public double sumQE=0.0;
	public ArrayList<Double> sumsQE=new ArrayList<Double>();
	public boolean displayGRN=true;

	protected GRNModel grn;

	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, GRNModel grn) {
		super(alpha, gamma, lambda, nbFeatures);
		this.grn = grn.copy();
	}

	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, Traces prototype, GRNModel grn, double alphaNorm) {
		super(alpha, gamma, lambda, nbFeatures, prototype);
		this.alphaNorm=alphaNorm;
		this.grn = grn.copy();	
	}

	public GRNSarsa(double alpha, double gamma, double lambda, PVector q, Traces prototype, GRNModel grn) {
		super(alpha, gamma, lambda, q, prototype);
		this.grn = grn.copy();	
	}

	@Override
	public double update(RealVector phi_t, RealVector phi_tp1, double r_tp1) {
		// updating parameters
		int smoothingSize=25;	
		step++;

//		double d=(delta-minDelta)/(maxDelta-minDelta);
//		if (sumsD.size()>smoothingSize) {
//			sumD-=sumsD.get(0);
//			sumsD.remove(0);
//			sumsD.add(d);
//			sumD+=d;
//		} else {
//			sumsD.add(d);
//			sumD+=d;
//		}
//		grn.proteins.get(0).concentration = 0.0;// Math.min(1.0, Math.max(0.0, sumD/sumsD.size()));
	
		double norm = q.dotProduct(q)*e.vect().dotProduct(e.vect());
		double qe=(q.dotProduct(e.vect())/(Math.abs(norm)<0.00001?0.00001:norm)+1.0)/2.0;
		double qeRate=0.4995;
		double qeModif = qe<qeRate?0:(qe-qeRate)/(0.5-qeRate);
//		System.out.println(qe+"\t"+qeModif);
		int div=0;
		sumQE=0;
		sumsQE.add(qeModif);
		if (sumsQE.size()>smoothingSize) {
			sumsQE.remove(0);
		}
		for (int i=0; i<sumsQE.size(); i++) {
			sumQE+=sumsQE.get(i)*(i+1);
			div+=(i+1);
		}
		grn.proteins.get(0).concentration = Math.min(1.0, Math.max(0.0, sumQE/div));

		double avgE=e.vect().sum()/e.vect().accessData().length;
		sumAvgE=0;
		div=0;
		avgsE.add(avgE);
		if (avgsE.size()>smoothingSize) {
			avgsE.remove(0);
		}
		for (int i=0; i<avgsE.size(); i++) {
			sumAvgE+=avgsE.get(i)*(i+1);
			div+=(i+1);
		}
		grn.proteins.get(1).concentration = 1.0-Math.min(1.0, Math.max(0.0, (sumAvgE/div)*(sumAvgE/div)*2500));

//		grn.proteins.get(2).concentration = Math.min(1.0, Math.max(0.0, step<stepMax?(0.5*Math.cos(((double)step)*Math.PI/stepMax)+0.5):0));
		grn.proteins.get(2).concentration = 1.0-Math.min(1.0, Math.max(0.0, Math.exp(-((double)(step*step))/((double)(stepMax*stepMax)))));
		
		grn.evolve(1);

//		if (grn.proteins.get(3).concentration>grn.proteins.get(9).concentration) {
			this.alpha=grn.proteins.get(4).concentration/(grn.proteins.get(4).concentration+grn.proteins.get(3).concentration+0.001)/alphaNorm;
			this.gamma=grn.proteins.get(5).concentration/(grn.proteins.get(5).concentration+grn.proteins.get(3).concentration+0.001);
			this.lambda=grn.proteins.get(6).concentration/(grn.proteins.get(6).concentration+grn.proteins.get(3).concentration+0.001);
//		}
			
		if (displayGRN) {
			for (int i=0; i<3; i++) {
				System.out.print(grn.proteins.get(i).concentration+"\t");
			}
			System.out.println(this.alpha+"\t"+this.gamma+"\t"+this.lambda+"\t"+episode);
		}
		
		return super.update(phi_t, phi_tp1, r_tp1);
	}

	@Override
	protected double initEpisode() {
		/**/		sumAvgE=0.0;
		avgsE=new ArrayList<Double>();
		sumD=0.0;
		sumsD=new ArrayList<Double>();
		sumQE=0.0;
		sumsQE=new ArrayList<Double>();/**/
		step=0;
		episode++;
		/*		grn.reset();
		grn.evolve(25);/**/
		return super.initEpisode();
	}

}
