import java.util.Random;

import javax.swing.*;

public class QLen_Agent {

    private static MountainCarEnv game;

    // Tiling configuration
    private static final int[] ACTIONS = { MountainCarEnv.REVERSE, MountainCarEnv.NOTHING, MountainCarEnv.FORWARD };
    private static final int NUM_ACTIONS = ACTIONS.length;
    private static final int OVERLAPPING_TILINGS = 8;
    private static final int CELLS_PER_DIMENSION = 8;
    private static final int CELLS_PER_TILING = CELLS_PER_DIMENSION * CELLS_PER_DIMENSION;
    private static final int TOTAL_TILES = CELLS_PER_TILING * OVERLAPPING_TILINGS;
    private static final int TOTAL_FEATURES = TOTAL_TILES * NUM_ACTIONS;

    private static final int EPISODES = 100000;
    private static final double GAMMA = 0.99;
    private static final double EPSILON = 0.1;
    private static final double ALPHA = 0.01 / OVERLAPPING_TILINGS;

    double[] w = new double[TOTAL_FEATURES]; // Weight for each feature (Tile)
    Random rng = new Random(42);

    public static void main(String[] args) {
        game = new MountainCarEnv(MountainCarEnv.NONE);

        // Execute episodes using the learned policy
        game = new MountainCarEnv(MountainCarEnv.NONE);

        QLen_Agent agent = new QLen_Agent();
        agent.train(QLen_Agent.EPISODES);

        agent.executePolicy(5, true);

        agent.createVisualizationData();
    }

    public void train(int numEpisodes) {

        for (int episode = 0; episode < numEpisodes; episode++) {

            double[] s = game.randomReset();
            int steps = 0;
            boolean done = false;

            while (!done) {

                // Compute the qValues
                double qValues[] = computeAllQ(s);

                // Pick an action (e-greedy)
                int a = chooseAction(qValues);

                double[] nextState = game.step(ACTIONS[a]);
                done = nextState[0] == 1;
                double reward = nextState[1];

                double target;
                if (done) {
                    target = reward;
                } else {
                    double[] qNext = computeAllQ(nextState);
                    target = reward + GAMMA * max(qNext);
                }

                double qCurrent = computeQ(s, a);
                double delta = target - qCurrent; // Delta error

                int[] activeTiles = getActiveTileIndices(s, a);
                for (int tileIdx : activeTiles) {
                    w[tileIdx] += ALPHA * delta;
                }

                s = nextState;
                steps++;
            }

            if (episode % 100 == 0) {
                System.out.println("Episode: " + episode + " finished in " + steps + " steps.");
            }
        }

    }

    /**
     * Compute Q(s, a, w) for ALL actions.
     * Returns array of size NUM_ACTIONS.
     */
    double[] computeAllQ(double[] s) {
        double[] qValues = new double[NUM_ACTIONS];
        for (int a = 0; a < NUM_ACTIONS; a++) {
            qValues[a] = computeQ(s, a);
        }
        return qValues;
    }

    /**
     * Compute Q(s, a, w) for a specific action.
     */
    double computeQ(double[] s, int a) {
        int[] activeTiles = getActiveTileIndices(s, a);
        double q = 0.0;
        for (int tileIdx : activeTiles) {
            q += w[tileIdx];
        }
        return q;
    }

    int[] getActiveTileIndices(double[] s, int a) {
        int[] stateTiles = getStateTileIndices(s);
        int[] fullIndices = new int[OVERLAPPING_TILINGS];
        // Don't understand
        int actionOffset = a * TOTAL_TILES;

        for (int i = 0; i < OVERLAPPING_TILINGS; i++) {
            fullIndices[i] = stateTiles[i] + actionOffset;
        }
        return fullIndices;
    }

    /**
     * action selection.
     *
     * With probability epsilon: pick a random action
     * With probability 1-epsilon: pick the best action
     */
    int chooseAction(double[] qValues) {
        if (rng.nextDouble() < EPSILON) {
            return rng.nextInt(NUM_ACTIONS); // random action
        } else {
            return argmax(qValues); // greedy action
        }
    }

    int[] getStateTileIndices(double[] s) {
        int[] result = new int[OVERLAPPING_TILINGS];

        // Normalise state variables to [0, 1]
        double posNorm = (s[2] - MountainCarEnv.MIN_POS) / (MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS);
        double speedNorm = (s[3] - MountainCarEnv.MIN_SPEED) / (MountainCarEnv.MAX_SPEED - MountainCarEnv.MIN_SPEED);

        for (int i = 0; i < OVERLAPPING_TILINGS; i++) {

            double tileSize = 1.0 / CELLS_PER_DIMENSION;
            double offset = i * tileSize / OVERLAPPING_TILINGS;

            double posOffsetted = posNorm + 1.0 * offset;
            double speedOffsetted = speedNorm + 2.0 * offset;

            // Identify the location of the tile in the grid (col, row)
            int col = clamp((int) (posOffsetted * CELLS_PER_DIMENSION), 0, CELLS_PER_DIMENSION - 1);
            int row = clamp((int) (speedOffsetted * CELLS_PER_DIMENSION), 0, CELLS_PER_DIMENSION - 1);

            // Convert the value to an integer useful to identify the weight
            result[i] = i * CELLS_PER_TILING + row * CELLS_PER_DIMENSION + col;
        }

        return result;
    }

    double max(double[] values) {
        return values[argmax(values)];
    }

    int argmax(double[] values) {
        int bestIdx = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[bestIdx])
                bestIdx = i;
        }
        return bestIdx;
    }

    int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    public void executePolicy(int numEpisodes, boolean visualize) {
        // Create a new environment with visualization if requested
        MountainCarEnv env = new MountainCarEnv(visualize ? MountainCarEnv.RENDER : MountainCarEnv.NONE);

        System.out.println("\nExecuting learned policy for " + numEpisodes + " episodes...\n");

        int successCount = 0;
        int totalSteps = 0;

        for (int episode = 0; episode < numEpisodes; episode++) {
            double[] s = env.randomReset();
            int steps = 0;
            double totalReward = 0;
            boolean done = false;

            while (!done) {
                // Computing the Q-values for all actions
                double[] qValues = computeAllQ(s);

                // Choosing the best action
                int a = argmax(qValues);

                // Taking the action
                double[] nextState = env.step(ACTIONS[a]);
                done = nextState[0] == 1;
                totalReward += nextState[1];

                s = nextState;
                steps++;

                if (steps > 1000) {
                    break;
                }
            }

            if (done) {
                successCount++;
                System.out.println("Episode " + (episode + 1) + " succeeded! Steps: " + steps +
                        ", Total Reward: " + totalReward);
            } else {
                System.out.println("Episode " + (episode + 1) + " failed (timeout).");
            }

            totalSteps += steps;
        }

        System.out.println("\nResults: " + successCount + "/" + numEpisodes + " successful");
        System.out.println("Average steps: " + (totalSteps / (double) numEpisodes));
    }

    public void createVisualizationData() {
        int gridSize = 25;
        double[][] vValues = new double[gridSize][gridSize];
        double[][] policy = new double[gridSize][gridSize];

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                // Map grid indices to actual position and velocity
                double position = MountainCarEnv.MIN_POS +
                        (i / (double) gridSize) * (MountainCarEnv.MAX_POS - MountainCarEnv.MIN_POS);
                double velocity = MountainCarEnv.MIN_SPEED +
                        (j / (double) gridSize) * (MountainCarEnv.MAX_SPEED - MountainCarEnv.MIN_SPEED);

                // Create state array [done, reward, position, velocity]
                double[] state = { 0, 0, position, velocity };

                // Compute Q-values for all actions at this state
                double[] qValues = computeAllQ(state);

                // V-value is the maximum Q-value
                vValues[i][j] = max(qValues);

                // Policy is the best action
                policy[i][j] = argmax(qValues);
            }
        }

        // Now display the heatmap
        try {
            HeatMapWindow hm = new HeatMapWindow(vValues, policy);
            hm.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            hm.setSize(600, 600);
            hm.setVisible(true);
            hm.update(vValues, policy);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

}
