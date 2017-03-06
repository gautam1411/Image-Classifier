/**
*
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

        if(debugOutputLayer) {
            System.out.println("<OutputLayer> : OutputLayer Constructor previous layer as FlatLayer ");
            System.out.println("<OutputLayer> : Number of classes : "+countClasses);

        }

        int countInputs = flat.getCountInputs();

        if(debugOutputLayer) {
            System.out.println(" input array size " + countInputs);

        }
        inputs = new Double [countInputs];
        output = new Double[countClasses];
        expected = new Double[countClasses];
        weights = new Double [countClasses][countInputs];
        initWeights();
    }

    public void initWeights(){

        Random rand = new Random();
        Random randSign = new Random();

        for(int i = 0; i < weights.length; i++){

            for(int j = 0; j < weights[0].length; j++){

                int sign = randSign.nextInt() % 2;
                weights[i][j] = rand.nextDouble();

                if( sign == 0)
                    weights[i][j] *= -1;

            }

        }

    }

    public void printPrediction(){

        for(int i = 0; i < countClasses; i++){
            System.out.println("  "+i+" : "+output[i]);
        }
    }

    public void readInputs(FlatLayer flat){

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

            for( int j = 0; j < weights[0].length; j++) {

                //System.out.println(" i,j : " + i+"   "+j);
                sum += weights[i][j] * inputs[j] ;
            }
            output[i] = sum;

        }

    }

}