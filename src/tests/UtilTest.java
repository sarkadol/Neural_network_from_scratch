package src.tests;

import src.Util;

import java.util.Arrays;

public class UtilTest {
    public static void main(String[] args) {

        Util util = new Util();
        System.out.println("ReLU");
        System.out.println(util.ReLU(-5));
        System.out.println(util.ReLU(0));
        System.out.println(util.ReLU(5));

        System.out.println("Labels to vectors");
        for(int i=0; i<10;i++){
            System.out.println(Arrays.toString(util.labelToVector(i)));
        }

        float[] pole = {1, 2, 3};
        //float[] result = util.Softmax(pole);
        System.out.println("softmax 1,2,3");
        //System.out.println(Arrays.toString(result)); //má být cca [0.0900,0.2447,0.6652] pro 1,2,3

    }

}
