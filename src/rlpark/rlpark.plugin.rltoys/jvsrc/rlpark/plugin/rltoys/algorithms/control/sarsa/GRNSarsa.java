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
	public double sumAvgE=0.0;
	public ArrayList<Double> avgsE=new ArrayList<Double>();
	public double sumD=0.0;
	public ArrayList<Double> sumsD=new ArrayList<Double>();
	public double sumQE=0.0;
	public ArrayList<Double> sumsQE=new ArrayList<Double>();
	
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
		/*if (delta>maxDelta || delta<minDelta) {
			System.err.println("Delta error : "+delta+"\t"+minDelta+"\t"+maxDelta);
		}*/
		//		if (delta!=delta) delta=minDelta;
		//		delta=Math.max(delta, minDelta);
		//		delta=Math.min(delta, maxDelta);
		
		//double norm = q.dotProduct(q)*e.vect().dotProduct(e.vect());
		//System.out.println((q.dotProduct(e.vect())/(Math.abs(norm)<0.00001?0.00001:norm)+1)/2.0);
		
//		double maxE=-Double.MAX_VALUE;
//		double minE=Double.MAX_VALUE;
//		for (double v : e.vect().accessData()) {
//			maxE=Math.max(maxE, v);
//			minE=Math.min(minE, v);
//		}
		
//		System.out.println(minE+"\t"+maxE+"\t"+e.vect().sum()/e.vect().accessData().length);
		int smoothingSize=25;		
		
		double d=(delta-minDelta)/(maxDelta-minDelta);
		if (sumsD.size()>smoothingSize) {
			sumD-=sumsD.get(0);
			sumsD.remove(0);
			sumsD.add(d);
			sumD+=d;
		} else {
			sumsD.add(d);
			sumD+=d;
		}
		grn.proteins.get(0).concentration = Math.min(1.0, Math.max(0.0, sumD/sumsD.size()));
		
		double norm = q.dotProduct(q)*e.vect().dotProduct(e.vect());
		double qe=(q.dotProduct(e.vect())/(Math.abs(norm)<0.00001?0.00001:norm)+1.0)/2.0;
		double qeModif = qe<0.4975?0:(qe-0.4975)/(0.5-0.4975);
		if (sumsQE.size()>smoothingSize) {
			sumQE-=sumsQE.get(0);
			sumsQE.remove(0);
			sumsQE.add(qeModif);
			sumQE+=qeModif;
		} else {
			sumsQE.add(qeModif);
			sumQE+=qeModif;
		}
		grn.proteins.get(1).concentration = Math.min(1.0, Math.max(0.0, sumQE/sumsQE.size()));
		
		double avgE=e.vect().sum()/e.vect().accessData().length;
		if (avgsE.size()>smoothingSize) {
			sumAvgE-=avgsE.get(0);
			avgsE.remove(0);
			avgsE.add(avgE);
			sumAvgE+=avgE;
		} else {
			avgsE.add(avgE);
			sumAvgE+=avgE;
		}
		grn.proteins.get(2).concentration = Math.min(1.0, Math.max(0.0, sumAvgE/avgsE.size()*50));
		grn.evolve(1);
		if (grn.currentStep%5==0) {
			this.alpha=grn.proteins.get(3).concentration/(grn.proteins.get(3).concentration+grn.proteins.get(6).concentration+0.001)/alphaNorm;
			this.gamma=grn.proteins.get(4).concentration/(grn.proteins.get(4).concentration+grn.proteins.get(6).concentration+0.001);
			this.lambda=grn.proteins.get(5).concentration/(grn.proteins.get(5).concentration+grn.proteins.get(6).concentration+0.001);
		}
		//		if (grn.currentStep%25==0) {
		//			this.alpha=grn.proteins.get(1).concentration/(grn.proteins.get(1).concentration+grn.proteins.get(4).concentration+0.001)/alphaNorm;
		//			this.gamma=grn.proteins.get(2).concentration/(grn.proteins.get(2).concentration+grn.proteins.get(4).concentration+0.001);
		//			this.lambda=grn.proteins.get(3).concentration/(grn.proteins.get(3).concentration+grn.proteins.get(4).concentration+0.001);
		//		}
/*		for (int i=0; i<3; i++) {
			System.out.print(grn.proteins.get(i).concentration+"\t");
		}
		System.out.println(this.alpha+"\t"+this.gamma+"\t"+this.lambda);
/**/		
		return super.update(phi_t, phi_tp1, r_tp1);
	}

	@Override
	protected double initEpisode() {
/**/		sumAvgE=0.0;
		avgsE=new ArrayList<Double>();
		sumD=0.0;
		sumsD=new ArrayList<Double>();
		sumQE=0.0;
		sumsQE=new ArrayList<Double>();
		grn.reset();
		grn.evolve(25);/**/
		return super.initEpisode();
	}

}
