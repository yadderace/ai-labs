import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.json.simple.*;
import org.json.simple.parser.*;

public class SmartAgent {

    private static BlackJackEnv game;
    private static ArrayList<String> gamestate;
    private static Map<String, double[]> qTable;

    public static void main(String[] args) {

        loadQTable("q_table.json");

        // Create game environment with rendering
        game = new BlackJackEnv(BlackJackEnv.RENDER);

        // Track statistics
        int totalGames = 20;
        int wins = 0;
        int losses = 0;
        int draws = 0;

        for (int i = 0; i < totalGames; i++) {
            gamestate = game.reset();
            System.out.println("=== Game " + (i + 1) + " ===");
            System.out.println("Initial state: " + gamestate);

            while (gamestate.get(0).equals("false")) { // Game is not over yet

                // Extract state information
                int playerSum = BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(gamestate));
                int dealerCard = BlackJackEnv.valueOf(BlackJackEnv.getDealerCards(gamestate).get(0));
                boolean usableAce = BlackJackEnv.holdActiveAce(BlackJackEnv.getPlayerCards(gamestate));

                System.out.println("Player sum: " + playerSum +
                        ", Dealer showing: " + dealerCard +
                        ", Usable ace: " + usableAce);

                // Choose action using Q-table
                int action = getBestAction(playerSum, dealerCard, usableAce);

                String actionName = (action == 0) ? "HIT" : "STAND";
                System.out.println("Q-Learning decision: " + actionName);

                // Execute action
                gamestate = game.step(action);
                System.out.println("Reward: " + gamestate.get(1));
            }

            int finalReward = Integer.parseInt(gamestate.get(1));
            System.out.println("  Dealer: " + BlackJackEnv.getDealerCards(gamestate) +
                    " (value: " + BlackJackEnv.totalValue(BlackJackEnv.getDealerCards(gamestate)) + ")");
            System.out.println("  Player: " + BlackJackEnv.getPlayerCards(gamestate) +
                    " (value: " + BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(gamestate)) + ")");

            if (finalReward > 0) {
                wins++;
            } else if (finalReward < 0) {
                losses++;
            } else {
                draws++;
            }
        }

        System.out.println("Wins:   " + wins + " (" + String.format("%.2f", 100.0 * wins / totalGames) + "%)");
        System.out.println("Losses: " + losses + " (" + String.format("%.2f", 100.0 * losses / totalGames) + "%)");
        System.out.println("Draws:  " + draws + " (" + String.format("%.2f", 100.0 * draws / totalGames) + "%)");
        System.out.println(
                "Win rate (wins + draws): " + String.format("%.2f", 100.0 * (wins + draws) / totalGames) + "%");
    }

    private static void loadQTable(String filename) {
        qTable = new HashMap<>();

        try {
            String content = new String(Files.readAllBytes(Paths.get(filename)));
            JSONParser parser = new JSONParser();
            JSONObject json = (JSONObject) parser.parse(content);

            for (Object key : json.keySet()) {
                String stateKey = (String) key;
                JSONArray values = (JSONArray) json.get(stateKey);

                // Extract Q-values for both actions (HIT=0, STAND=1)
                double[] qValues = new double[2];
                qValues[0] = ((Number) values.get(0)).doubleValue();
                qValues[1] = ((Number) values.get(1)).doubleValue();

                qTable.put(stateKey, qValues);
            }

            System.out.println("Q-table loaded successfully from " + filename);
            System.out.println("Total states in Q-table: " + qTable.size());

        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            System.exit(1);
        } catch (ParseException e) {
            System.err.println("Error parsing JSON: " + e.getMessage());
            System.exit(1);
        }
    }

    private static int getBestAction(int playerSum, int dealerCard, boolean usableAce) {
        // Convert Ace from 11 to 1 to match QTable format
        if (dealerCard == 11) {
            dealerCard = 1;
        }

        String stateKey = playerSum + "," + dealerCard + "," + (usableAce ? "1" : "0");
        double[] qValues = qTable.get(stateKey);

        if (qValues == null) {
            System.err.println("  [State not in Q-table, using default strategy]");
            System.err.println("  [Missing state: " + stateKey + "]");
            System.exit(1);
        }
        int bestAction = (qValues[0] > qValues[1]) ? BlackJackEnv.STAND : BlackJackEnv.HIT;

        System.out.println("  [Q-values: HIT=" + String.format("%.2f", qValues[0]) +
                ", STAND=" + String.format("%.2f", qValues[1]) + "]");

        return bestAction;
    }
}
