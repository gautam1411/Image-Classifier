/**
 *
 *
 * This class defines Output Layer
 *
 */

import java.util.Random;

public class OutputLayer{


    private Double [] inputs;
    private Double [] output;
    private Double [] expected;
    private Double [][] weights;
    private Double [][] oldWeights;
    private Double [] errors;
    private int countCorrect;
    private int confusionMatrix [][];//rows = predicted and cols = actual

    private boolean debugOutputLayer = false;
    private int countClasses;

    private Double label;


    public OutputLayer(){

        if(debugOutputLayer)
        System.out.println("<OutputLayer> : OutputLayer Default Constructor");
        output = new Double[countClasses];
        expected = new Double[countClasses];
    }

    public OutputLayer(FlatLayer flat, int hyperparameters, boolean debugSwitch){

        debugOutputLayer = debugSwitch;
        countClasses = (hyperparameters >> 24)& (0XF);

        int countInputs = flat.getCountOutputs();
        
        countCorrect = 0;

        if(debugOutputLayer) {
            System.out.println("<OutputLayer> : OutputLayer Constructor previous layer as FlatLayer ");
            System.out.println("<OutputLayer> : Number of classes : "+countClasses);
            System.out.println(" input array size " + countInputs);

        }

        if( countInputs <= 0 || countClasses <= 0){
            System.out.println("<OutputLayer> : Inavlid parameter");
        }else {
            inputs = new Double[countInputs+1];
            confusionMatrix = new int [countClasses][countClasses];
            output = new Double[countClasses];
            expected = new Double[countClasses];
            weights = new Double[countClasses][countInputs + 1]; // all inputs + bias
            oldWeights = new Double[countClasses][countInputs + 1]; // all inputs + bias
            errors = new Double[countClasses];
            initWeights();
        }
    }

    public void initWeights(){

        Random rand = new Random(2);
        Random randSign = new Random(2);

        for(int i = 0; i < weights.length; i++){

            for(int j = 0; j < weights[0].length; j++){

                int sign = randSign.nextInt() % 2;
                weights[i][j] =   rand.nextDouble();

                if( sign == 0)
                    weights[i][j] *= -1;

            }

        }

    }

    public void printPrediction(){

    	int index = 0;
    	double bestVal = 0.0;
    	
        if(output[Double.valueOf(label).intValue()].doubleValue() > 0.5){
        	countCorrect++;
        	confusionMatrix[Double.valueOf(label).intValue()][Double.valueOf(label).intValue()]++;
        	return;
        }
        for(int i = 0; i < countClasses; i++){
            if(output[i].doubleValue() > bestVal){
            	index = i;
            	bestVal = output[i].doubleValue();
            }
        }
        confusionMatrix[index][Double.valueOf(label).intValue()]++;

       // System.out.println("Input image label : "+label+ " Predicted Label " +Double.valueOf(label).intValue() + " In output: " + output[Double.valueOf(label).intValue()]);
        // System.out.println("");
    }

    public void readInputs(FlatLayer flat){

        this.label = flat.getLabel();
        Double [] ip = flat.getOutput();

        for (int i = 0; i < inputs.length-1; i++) {

            inputs[i] = ip[i];
            //System.out.println(" index: "+i+"   :  "+inputs[i]);
        }
        inputs[inputs.length-1] = 1.0; //bias unit
        
        if(debugOutputLayer)
        System.out.println("<OutputLayer> Input read successfully : " + inputs.length);

    }

    public void train(FlatLayer flat){

        readInputs( flat);

        for (int i = 0; i< countClasses ; i++){

            Double sum = 0.0;

            for( int j = 0; j < weights[0].length-1; j++) {

               // System.out.println(" i,j : " + i+"   "+j);
                sum += weights[i][j] * inputs[j] ;
            }
            output[i] = sum;

        }

        //softmax(output);
        sigmoid(output);
        //printPrediction();

    }
    
    
    
    public void sigmoid( Double output[] ){
    	
    	 for(int i =0 ; i < output.length; i++){
             output[i] = ( 1 / ( 1 + Math.exp( -output[i] )) );
         }
    }

    public int getCountCorrect(){
    	
    	return countCorrect;
    }
    
    public void resetCountCorrect(){
    	countCorrect = 0;
    }
    
    public void softmax( Double output[] ){

        Double sum = 0.0;

        for(int i =0 ; i < output.length; i++){
            Double exp_val = Math.exp(output[i]);
            sum += exp_val;
            output[i] = exp_val;
        }

        for(int i = 0; i < output.length ; i++){
            output[i] /= sum;
        }
    }

    public void backpropagate( ){

        // Assume  ==> learning rate =  0.01
        Double learning_rate = 0.001;

        int predicted_index = ( Double.valueOf(label) ).intValue( ) ;
        expected[predicted_index] = 1.0;

        for(int i = 0 ; i < errors.length; i++) {
            errors[i] = 0.0;
        }

        for( int i = 0 ; i < output.length; i++ ){

        	//should weights add the delta or subtract the delta/error?
            if( i == predicted_index){
               errors[i] += ( 1 - output[i] );
            }else{
               errors[i] +=  ( 0 - output[i] );
            }

            for(int j = 0; j < inputs.length; j++){

            	//should weights add the delta or subtract the delta/error?
            	oldWeights[i][j] = weights[i][j];
                weights[i][j] += errors[i] * learning_rate * inputs[j];
            }
        }
    }
    
    public Double [][] getWeights(){
    	
    	return weights;
    }
    
    public Double [][] getOldWeights(){
    	return oldWeights;
    }
    
    public void zeroConfusionMatrix(){
    	
    	for(int i = 0; i < confusionMatrix.length; i++){
    		for(int j = 0; j < confusionMatrix[0].length; j++){
    			confusionMatrix[i][j] = 0;
    		}
    	}
    }
    
    public int [][] getConfusionMatrix(){
    	return confusionMatrix;
    }
    
    public void printConfusion(){
    	for(int i = 0; i < confusionMatrix.length; i++){
    		for(int j = 0; j < confusionMatrix[0].length; j++){
    			System.out.print(confusionMatrix[i][j]);
    			System.out.print(" ");
    		}
    		System.out.println("");
    	}
    }

    public int reportPredictionError(){

        int predicted_index = ( Double.valueOf(label) ).intValue( ) ;
        int max_index = -1;
        Double max_score = Double.MIN_VALUE;
        for(int i = 0; i < countClasses; i++){

            if(output[i] > max_score){
                max_score = output[i];
                max_index = i;
            }

        }

        return (max_index == predicted_index) ? 0: 1 ;
    }

    public Double [] getErrors(){

        return errors;
    }

}