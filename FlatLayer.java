/**
 *
 *
 *
 * This class defines fully connected(flat) layer in ConvNet
 *
 */

import java.util.ArrayList;
//import java.lang.System;

public class FlatLayer{


    private Double [] input;

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

        readInputFeature(poolMaps);

    }

    public FlatLayer( Pooling pool, boolean debugSwitch){

        debugFlatLayer = debugSwitch;

        int   countPoolMap    = pool.countPoolMaps();
        int  dimPlate         = pool.outputVolume();

        countInputs = (countPoolMap) *(dimPlate*dimPlate);

        input = new Double[countInputs];

        if(debugFlatLayer)
        System.out.println(" <FlatLayer>: Constructor  for Pool as previous layer  : " + countInputs);

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

}


