import javax.swing.JFrame;

public class StateDiscretizationTester {
    // This class can help you test your state-space discretization
    private static final int POS_BINS = 400;
    private static final int VEL_BINS = 400;
    private static double[][] valueTable = new double[POS_BINS][VEL_BINS];
    private static final double MIN_POS = -1.2;
    private static final double MAX_POS = 0.6;
    private static final double MIN_SPEED = -0.07;
    private static final double MAX_SPEED = 0.07;

    private static int discretizePosition(double position) {
        // Maps from MIN_POS to MAX_POS to 0 to POS_BINS-1
        double normalized = (position - MIN_POS) /
                (MAX_POS - MIN_POS);
        int bin = (int) (normalized * POS_BINS);
        return bin;
    }

    private static int discretizeVelocity(double velocity) {
        // Maps from MIN_SPEED to MAX_SPEED to 0 to VEL_BINS-1
        double normalized = (velocity - MIN_SPEED) /
                (MAX_SPEED - MIN_SPEED);
        int bin = (int) (normalized * VEL_BINS);
        return bin;
    }

    // The following two methods should store and retrieve Values for a given
    // position and velocity
    private static double getValue(double position, double velocity) {
        // to be filled by you with your discretization method
        int posBin = discretizePosition(position);
        int velBin = discretizeVelocity(velocity);
        return valueTable[posBin][velBin];
    }

    private static void putValue(double position, double velocity, double value) {
        // to be filled by you with your discretization method
        int posBin = discretizePosition(position);
        int velBin = discretizeVelocity(velocity);
        valueTable[posBin][velBin] = value;
    }

    ///////////////////////////////////////////
    // Don't change any code below this line //
    ///////////////////////////////////////////

    private static final int nb = 1000;

    public static void main(String[] args) {

        try {
            // This first part will show the original values
            double[][] originalValues = new double[1000][1000];
            for (int i = 0; i < 1000; i++)
                for (int j = 0; j < 1000; j++)
                    originalValues[i][j] = Math.sin(0.00002 * i * j);
            HeatMapWindow hm1 = new HeatMapWindow(originalValues);
            hm1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            hm1.setSize(1000, 1000);
            hm1.setVisible(true);
            hm1.update(originalValues);

            // This second part will show the values you stored
            // If all is well, both images should like alike, possibly
            // with a slight reduction in resolution
            double[][] storedValues = new double[1000][1000];
            for (int i = 0; i < 1000; i++)
                for (int j = 0; j < 1000; j++)
                    putValue(toPosition(i), toVelocity(j), Math.sin(0.00002 * i * j));
            for (int i = 0; i < 1000; i++)
                for (int j = 0; j < 1000; j++)
                    storedValues[i][j] = getValue(toPosition(i), toVelocity(j));

            HeatMapWindow hm2 = new HeatMapWindow(storedValues);
            hm2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            hm2.setSize(1000, 1000);
            hm2.setVisible(true);
            hm2.update(storedValues);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private static double toPosition(int x) {
        return 1.0 * x / nb * (MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS) + MountainCarEnv.MIN_POS
                + (MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS) / (nb * 2);
    }

    private static double toVelocity(int x) {
        return 1.0 * x / nb * (MountainCarEnv.MAX_SPEED * 2) - MountainCarEnv.MAX_SPEED + MountainCarEnv.MAX_SPEED / nb;
    }
}
