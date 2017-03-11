/**
 *
 *
 *
 * This class defines fully connected(flat) layer in ConvNet
 *
 */

import java.util.ArrayList;
import java.util.Random;

public class FlatLayer{


    private Double [] input;
    private Double [] errors;
    private Double [] output;
    private Double [][] weights;
    private Double [][] oldWeights;

    private int countInputs;
    private int countOutputs;

    private Double label;

    private boolean debugFlatLayer = true;

    public FlatLayer(){

        System.out.println(" <FlatLayer>: Default Constructor");

    }

    public FlatLayer(Convolution conv){

        System.out.println(" <FlatLayer>: Constructor  for Conv as previous layer");

       // int dimPlate = conv.outputVolume();
       // int sizePlate = dimPlate * dimPlate ;

        

    }

    public  void readInputFeature(  ArrayList<PoolMap> poolMaps ){

        if(debugFlatLayer)
        System.out.println(" PoolMaps size is : " + poolMaps.size());


        for(int i = 0; i <  poolMaps.size(); i++){

            PoolMap plate = poolMaps.get(i);

            int dimPlate = plate.getOutVol();
            int sizePlate = dimPlate * dimPlate ;
            Double [][] plateOut  = plate.getOutput();

            //System.out.println( " DimPlate : "+dimPlate+"  SizePlate " + sizePlate + "  plateOut.length: " + plateOut.length);

            for( int j = 0 ; j < plateOut.length; j++){
            	
                System.arraycopy(plateOut[j],0, input, (i*sizePlate)  + (j * dimPlate) , plateOut[j].length);
            }

        }
        
        input[input.length-1] = 1.0; //bias unit;
    }
    
    public  void train(Pooling pool){

        ArrayList<PoolMap> poolMaps = pool.get_P_maps();
        this.label = pool.getLabel();
        readInputFeature(poolMaps);
        

        for(int j = 0; j < weights.length; j++){
        	
        	output[j] = 0.0;
        		
        	for(int i = 0; i < input.length; i++){

        		output[j] += weights[j][i]*input[i];
        	}
        }
        
        calcOutputs(output);
        //calcOutputsWithDropout(output);

    }

    public  void trainwithDropOut(Pooling pool){

        ArrayList<PoolMap> poolMaps = pool.get_P_maps();
        this.label = pool.getLabel();
        readInputFeature(poolMaps);


        for(int j = 0; j < weights.length; j++){

            output[j] = 0.0;

            for(int i = 0; i < input.length; i++){

                output[j] += weights[j][i]*input[i];
            }
        }

        // calcOutputs(output);
        calcOutputsWithDropout(output);

    }
    
    public void calcOutputs(Double [] out){
    	
    	for(int i = 0; i < out.length; i++){
    		out[i] = LeakyReLU(out[i]);
    	}
    }

    public void calcOutputsWithDropout(Double [] out){

        Random rand = new Random();

        for(int i = 0; i < out.length; i++){

            out[i] = LeakyReLU(out[i]);

            //int val = rand.nextInt() % 10;

            //if(val ==0){
            //    out[i] = 0.0; // dropout
           // }

        }
    }
    
    public Double ReLU(Double  out){
    	
    	return out = ((out>= 0) ? out : 0);
    		
    }

    public Double LeakyReLU(Double  out){
    	
    	return out = ((out>= 0) ? out : 0.1*out);

    }
    
    public FlatLayer( Pooling pool, boolean debugSwitch){

        debugFlatLayer = debugSwitch;

        int   countPoolMap    = pool.countPoolMaps();
        int  dimPlate         = pool.outputVolume();
        int numHiddenUnits = 250;

        this.label = pool.getLabel();
        //countInputs = (countPoolMap) *(dimPlate*dimPlate);
        
        countInputs = countPoolMap*dimPlate*dimPlate;
        countOutputs = numHiddenUnits;

        input = new Double[countInputs+1];//plus one for bias
        errors = new Double[numHiddenUnits];
        weights = new Double[numHiddenUnits][countInputs+1]; //plus one for bias
        oldWeights = new Double[numHiddenUnits][countInputs+1]; //plus one for bias
        output = new Double[countOutputs];

        initWeights();
        
        if(debugFlatLayer)
        System.out.println(" <FlatLayer>: Constructor  for Pool as previous layer  : " + countInputs);

    }
    
    public void initWeights(){

        Random rand = new Random(1);
        Random randSign = new Random(1);

        for(int i = 0; i < weights.length; i++){

            for(int j = 0; j < weights[0].length; j++){

                int sign = randSign.nextInt() % 2;
                weights[i][j] =   rand.nextDouble();

                if( sign == 0)
                    weights[i][j] *= -1;

            }
        }
    }

    public int getCountInputs(){

        return countInputs;
    }
    
    public int getCountOutputs(){
    	
    	return countOutputs;
    }

    public void printAct(){

        for(int i =0 ; i < input.length; i++)
            System.out.println(" FlatLayer : "+i+"  "+input[i]);
    }

    public Double [] getInput (){

        return input;
    }
    
    public Double [] getOutput(){
    	
    	return output;
    }

    public Double getLabel (){

        return this.label;
    }

    public void backpropagate (OutputLayer out){

        Double [] err = out.getErrors();
        Double [][] outWeights = out.getOldWeights();
        Double learningRate = 0.001;
        
        //use leaky ReLU derivative is 1 or 0.1
        //update rule = 1*hiddenUnit input* (sum ( out errors* weight from hidden to out))
        Double sum; 
        
        for(int i = 0; i < countOutputs; i++){
        	
        	sum = 0.0;
        	
        	for(int j = 0; j < outWeights.length; j++){
        		
        		sum += outWeights[j][i] * err[j] ;
        	}
        	if(output[i] > 0){
        		errors[i] = sum; //TODO not sure if we want to switch to leaky ReLU to prevent 0 here
        	}else{
        		errors[i] = 0.1*sum;
        	}
        		
        }
        
        for(int i = 0; i<weights.length; i ++){
        	for(int j = 0; j<weights[0].length; j++ ){
        		oldWeights[i][j] = weights[i][j];
        		weights[i][j] += errors[i]*input[j]*learningRate;
        	}
        }
    }

    public Double [] getErrors(){

        return  errors;
    }
    
    public Double [][] getOldWeights(){

    	return oldWeights;
    }

}


