/***********************************************************************************************************************
 *
 *
 * This class defines fully connected(flat) layer in ConvNet
 *
 **********************************************************************************************************************/

import java.util.ArrayList;

public class FlatLayer{


    private Double [] input;
    private Double [] errors;

    private int countInputs;

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

    }


    public  void train(Pooling pool){

        ArrayList<PoolMap> poolMaps = pool.get_P_maps();
        this.label = pool.getLabel();
        readInputFeature(poolMaps);

    }

    public FlatLayer( Pooling pool, boolean debugSwitch){

        debugFlatLayer = debugSwitch;

        int   countPoolMap    = pool.countPoolMaps();
        int  dimPlate         = pool.outputVolume();

        countInputs = (countPoolMap) *(dimPlate*dimPlate);

        input = new Double[countInputs];
        errors = new Double[countInputs+1];

        if(debugFlatLayer) {
            System.out.println(" <FlatLayer>: Constructor  for Pool as previous layer  : " + countInputs);
        }

    }

    public int getCountInputs(){

        return countInputs;
    }

    public void printAct(){

        for(int i =0 ; i < input.length; i++)
            System.out.println(" FlatLayer : "+i+"  "+input[i]);
    }

    public Double [] getInput (){

        return input;
    }

    public Double getLabel (){

        return this.label;
    }

    public void backpropagate (OutputLayer out){

        Double [] err = out.getErrors();
        //this.errors = new Double[ err.length];

        System.arraycopy(err,0,errors,0,err.length);

        if(debugFlatLayer){

            System.out.println("<FlatLayer.java : backpropagate> ");

        }
    }

    public Double [] getErrors(){
        return  errors;
    }

}


