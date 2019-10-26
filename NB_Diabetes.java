import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.lang.Exception;

public class NB_Diabetes{

	public NB_Diabetes(){
		try{
			
			BufferedReader dataReader = new BufferedReader(new FileReader("/home/apoorva/Downloads/diabetes.arff"));

			Instances trainingSet = new Instances(dataReader);
			
			trainingSet.randomize(new java.util.Random(0));
			
			int trainSize = (int) Math.round(trainingSet.numInstances() * 0.8);
			int testSize = trainingSet.numInstances() - trainSize;
			
			Instances train = new Instances(trainingSet, 0, trainSize);
			Instances test = new Instances(trainingSet, trainSize, testSize);
			
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);
			
			NaiveBayes model=new NaiveBayes();
			model.buildClassifier(train);
			
			Evaluation eTest = new Evaluation(test);
			eTest.evaluateModel(model, test);
			String[] cmarray = {"tested_positive","tested_negative"};
			ConfusionMatrix cm = new ConfusionMatrix(cmarray);
			
			for (int i = 0; i < test.numInstances(); i++)
			{
				test.instance(i).setClassMissing();
				double cls = model.classifyInstance(test.instance(i));
				test.instance(i).setClassValue(cls);
			}
			System.out.println("Error Rate: "+eTest.errorRate()*100);
			System.out.println("Pct Correct: "+eTest.pctCorrect());
			for (int i=0; i<train.numClasses(); i++){
				System.out.println("Class "+ i);
				System.out.println("	Precision " +eTest.precision(i));
				System.out.println("	Recall "+eTest.recall(i));
				System.out.println("	Area under ROC "+eTest.areaUnderROC(i));
				System.out.println();
			}
		}
		catch (Exception o)
		{
			System.err.println(o.getMessage());
		}
	}
	
	public static void main(String[] args) {
		NB_Diabetes nb = new NB_Diabetes();
	}
}
