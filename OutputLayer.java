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
    private Double [] errors;

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

        int countInputs = flat.getCountInputs();

        if(debugOutputLayer) {
            System.out.println("<OutputLayer> : OutputLayer Constructor previous layer as FlatLayer ");
            System.out.println("<OutputLayer> : Number of classes : "+countClasses);
            System.out.println(" input array size " + countInputs);

        }

        if( countInputs <= 0 || countClasses <= 0){
            System.out.println("<OutputLayer> : Inavlid parameter");
        }else {
            inputs = new Double[countInputs];
            output = new Double[countClasses];
            expected = new Double[countClasses];
            weights = new Double[countClasses][countInputs + 1]; // all inputs + bias
            errors = new Double[countInputs + 1];
            initWeights();
        }
    }

    public void initWeights(){

        Random rand = new Random();
        Random randSign = new Random();

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

        for(int i = 0; i < countClasses; i++){
            System.out.println("  "+i+" : "+output[i]);
        }
        System.out.println("");
        System.out.println("Input image label : "+label);
    }

    public void readInputs(FlatLayer flat){

        this.label = flat.getLabel();
        Double [] ip = flat.getInput();

        for (int i = 0; i < inputs.length; i++) {

            inputs[i] = ip[i];
            //System.out.println(" index: "+i+"   :  "+inputs[i]);
        }
        if(debugOutputLayer)
        System.out.println("<OutputLayer> Input read successfully : " + inputs.length);

    }


    public void train(FlatLayer flat){

        readInputs( flat);

        for (int i = 0; i< countClasses ; i++){

            Double sum = 0.0;

            for( int j = 0; j < weights[0].length-1; j++) {

                System.out.println(" i,j : " + i+"   "+j);
                sum += weights[i][j] * inputs[j] ;
            }
            output[i] = sum;

        }

        softmax(output);
        printPrediction();

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

        // Assume  ==> learning rate =  0
        Double learning_rate = 0.01;

        int predicted_index = ( Double.valueOf(label) ).intValue( ) ;
        expected[predicted_index] = 1.0;

        for(int i = 0 ; i < errors.length; i++) {
            errors[i] = 0.0;
        }


        for( int i = 0 ; i < inputs.length; i++ ){

            errors[i] = 0.0;

            for(int j = 0; j < output.length; j++){


                if( j == predicted_index){

                   weights[j][i] += ((1- output[j]) *  (output[j]) ) * learning_rate * inputs[i];
                   //if(true)
                   //    System.out.println("<OutLayer> i  , j  : " +i+"   "+j+  "  "+errors[i]);
                   //System.out.println(" "+errors.length+ "  "+output.length+  "  " +inputs.length);
                   errors[i] += ((1- output[j]) *  (output[j]) );


                }else{

                    weights[j][i] += ((0- output[j]) *  (output[j]) ) * learning_rate * inputs[i];

                    //if(true)
                    //    System.out.println("<OutLayer> i  , j  : " +i+"   "+j+ "   "+errors[j]);
                   // System.out.println(" "+errors.length+ "  "+output.length+  "  " +inputs.length);
                    errors[i] +=  ((0- output[j]) *  (output[j]) );
                }
            }
        }

    }

    public Double [] getErrors(){

        return errors;
    }

}