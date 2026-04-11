import javax.swing.*;
import java.util.Arrays;

public class DPAgent {

    private static MountainCarEnv game;
    private static double[] gamestate;
    private static double[][] vValues;
    private static int[][] policy;

    private static final int POS_BINS = 400;
    private static final int VEL_BINS = 400;

    private static final int ITERATIONS = 1000;
    private static final double GAMMA = 0.99;
    private static final double THETA = 0.001;
    private static final int[] ACTIONS = { MountainCarEnv.REVERSE, MountainCarEnv.NOTHING, MountainCarEnv.FORWARD };

    /**
     * public static void main(String[] args) {
     * game = new MountainCarEnv(MountainCarEnv.RENDER);
     * // Running 100 episodes
     * for (int i = 0; i < 100; i++) {
     * gamestate = game.randomReset();
     * System.out.println("The initial gamestate is: " +
     * Arrays.toString(gamestate));
     * while (gamestate[0] == 0) { // Game is not over yet
     * System.out.println("The car's position is " + gamestate[2]);
     * System.out.println("The car's velocity is " + gamestate[3]);
     * if (gamestate[3] >= 0.0) {
     * System.out.println("I will try to go further forward");
     * gamestate = game.step(MountainCarEnv.FORWARD);
     * } else if (gamestate[3] < 0.0) {
     * System.out.println("I will try to continue going backwards");
     * gamestate = game.step(MountainCarEnv.REVERSE);
     * }
     * System.out.println("The gamestate passed back to me was: " +
     * Arrays.toString(gamestate));
     * System.out.println("I received a reward of " + gamestate[1]);
     * }
     * System.out.println();
     * }
     * try {
     * double[][] valuesToShow = new double[1000][1000];
     * for (int i = 0; i < 1000; i++)
     * for (int j = 0; j < 1000; j++)
     * valuesToShow[i][j] = Math.sin(0.00002 * i * j);
     * HeatMapWindow hm = new HeatMapWindow(valuesToShow);
     * hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
     * hm.setSize(600, 600);
     * hm.setVisible(true);
     * hm.update(valuesToShow);
     * } catch (Exception e) {
     * System.out.println(e.getMessage());
     * }
     * }
     * 
     **/

    public static void main(String[] args) {
        game = new MountainCarEnv(MountainCarEnv.NONE);
        vValues = new double[POS_BINS][VEL_BINS];
        policy = new int[POS_BINS][VEL_BINS];
        gamestate = game.randomReset();
        valueIteration();

        try {
            HeatMapWindow hm = new HeatMapWindow(vValues);
            hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            hm.setSize(600, 600);
            hm.setVisible(true);
            hm.update(vValues);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public static void valueIteration() {
        int iteration = 0;
        double delta = 0;
        do {
            delta = 0;
            iteration++;

            for (int i = 0; i < POS_BINS; i++) {
                for (int j = 0; j < VEL_BINS; j++) {
                    double position = indexToPosition(i);
                    double velocity = indexToVelocity(j);

                    if (position >= MountainCarEnv.GOAL_POS)
                        continue;
                    double oldValue = vValues[i][j];
                    double maxValue = Double.NEGATIVE_INFINITY;

                    for (int action : ACTIONS) {
                        double newValue = getExpectedValue(position, velocity, action);
                        if (newValue > maxValue) {
                            maxValue = newValue;
                            policy[i][j] = action;
                        }
                    }
                    vValues[i][j] = maxValue;
                    delta = Math.max(delta, Math.abs(maxValue - oldValue));
                }
            }

            if (iteration % 10 == 0) {
                System.out.println("Iteration " + iteration + ": delta = " + delta);
            }

        } while (delta > THETA && iteration < ITERATIONS);

        System.out.println("Value Iteration converged after " + iteration + " iterations.");

    }

    private static double getExpectedValue(double position, double velocity, int action) {
        game.setState(position, velocity);
        double[] nextState = game.step(action);
        double reward = nextState[1];
        boolean goalReached = nextState[0] == 1;

        if (goalReached)
            return reward;

        double nextPosition = nextState[2];
        double nextVelocity = nextState[3];

        int nextPosBin = discretizePosition(nextPosition);
        int nextVelBin = discretizeVelocity(nextVelocity);
        return reward + GAMMA * vValues[nextPosBin][nextVelBin];
    }

    private static int discretizePosition(double position) {
        position = Math.max(MountainCarEnv.MIN_POS, Math.min(MountainCarEnv.MAX_POS - 1e-9, position));
        double bin_width = (MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS) / POS_BINS;
        int bin = (int) Math.floor((position - MountainCarEnv.MIN_POS) / bin_width);
        return bin;
    }

    private static int discretizeVelocity(double velocity) {
        velocity = Math.max(MountainCarEnv.MIN_SPEED, Math.min(MountainCarEnv.MAX_SPEED - 1e-9, velocity));
        double bin_width = (MountainCarEnv.MAX_SPEED - MountainCarEnv.MIN_SPEED) / VEL_BINS;
        int bin = (int) Math.floor((velocity - MountainCarEnv.MIN_SPEED) / bin_width);
        return bin;
    }

    private static double indexToPosition(int idx) {
        return MountainCarEnv.MIN_POS + ((MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS) / POS_BINS) * idx;
    }

    private static double indexToVelocity(int idx) {
        return MountainCarEnv.MIN_SPEED + ((MountainCarEnv.MAX_SPEED - MountainCarEnv.MIN_SPEED) / VEL_BINS) * idx;
    }
}
