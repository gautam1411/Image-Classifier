/***
 *
 *
 *
 * Pooling class defines all functionality at pooling layer
 *
 */

import java.util.ArrayList;
import java.util.Vector;
import java.util.List;

public class Pooling{

        // Boolean switch for debugging pooling layer
        private boolean debugPool = false;

        // Pooling layer plates
        private ArrayList<PoolMap> poolMaps;
        // Number of plates in pooling player
        private int countPoolMaps;
        // input size
        private int plateSize;
        // output plate size
        private int outVol;

        private Double label;



        public Pooling(){

            if(debugPool)
                System.out.println("Pooling layer default constructor");
            poolMaps = new ArrayList<PoolMap>();
        }


    public Pooling(Convolution conv, boolean debugSwitch){


        debugPool = debugSwitch;


        poolMaps = new ArrayList<PoolMap>();

        plateSize =  conv.outputVol;
        countPoolMaps = conv.countFeatureMaps;
        outVol = plateSize/2;

        if(debugPool) {
            System.out.println("<Pooling>: Parameterized  Constructor");
            System.out.println("<Pooling>: No of plates: " + countPoolMaps );
            System.out.println("<Pooling>: Input plate size: "+plateSize);
            System.out.println("<Pooling>: Output plate size: " + outVol);


        }

        for( int i = 0; i<countPoolMaps; i++) {

            PoolMap poolMap = new PoolMap(plateSize,outVol,debugSwitch);
            addPoolMap(poolMap);
        }

    }


    // Get number of Pool maps in the convolution layer
    public int countPoolMaps(){

        return poolMaps.size();
    }

    // Add a Pool map in the pooling layer
    public void addPoolMap(PoolMap poolMap){

        poolMaps.add(poolMap);
    }

    // Returns list of feature maps in the convolution layer
    public ArrayList<PoolMap> get_P_maps(){

        return poolMaps;
    }



    public void calcPoolMaps( ){

        if(debugPool)
        System.out.println(" <PoolingLayer> : calcPoolMaps");

        for(PoolMap pool_map: poolMaps)
            pool_map.computePoolMap(); // stride, padding

    }

    public void printPoolMaps(){

        for (PoolMap pool_map: poolMaps){

            pool_map.printPoolMap();

        }

    }

    public void readInputFeature(ArrayList<FeatureMap> feature_maps ){

        // Currently using Nth ConvLayer Plate mapped to Nth PoolLayer plate
        // We can  use multiple pooling plate of difeerent size corresponding
        // to each ConvLayer plate of previous layer

        if(debugPool)
            System.out.println(" No of featureMaps in previos ConvLayer : " + feature_maps.size());

       // for (FeatureMap feature_map: feature_maps){

       // }

        for (int i = 0; i <  feature_maps.size() ; i++){

            FeatureMap feature_map = feature_maps.get(i);

            Double [][]fMap = feature_map.getFeatureMap();

            PoolMap poolMap = poolMaps.get(i);

            Double [][] pMap = poolMap.getInputFeature();

            for(int j = 0 ; j < fMap.length; j++){

                System.arraycopy( fMap[j],0, pMap[j], 0, fMap[j].length);
            }
        }

    }

    public void train(  ArrayList<FeatureMap> feature_maps   ){

        readInputFeature(feature_maps);
        calcPoolMaps();

    }

    public int outputVolume(){

        return outVol;
    }

    public void  connectPreConv( Convolution conv){

    }

}