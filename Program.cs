using System;
using Keras;
using Tensorflow;
using Tensorflow.Keras;
using MathNet.Numerics;
using OneOf.Types;
using System.Runtime.InteropServices;
using Google.Protobuf.WellKnownTypes;

// Example Board Setups to copy

// 10 X 10 BOARD

// {8,8,8,8,8,8,8,8,8,8,8,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,0,0,0,0,0,0,0,0,0,0,8},
// {8,8,8,8,8,8,8,8,8,8,8,8}

// 5 X 5 BOARD

// {8,8,8,8,8,8,8},
// {8,0,0,0,0,0,8},
// {8,0,0,0,0,0,8},
// {8,0,0,0,0,0,8},
// {8,0,0,0,0,0,8},
// {8,0,0,0,0,0,8},
// {8,8,8,8,8,8,8}

namespace VRAG
{
    class VRAG
    {
        public static  NeuralNetWork trainedModel;

        static Random rnd;

        static void Main()
        {
            // We set the board that we will use
            int[,] rootBoard = new int[,] {
                {8,8,8,8,8,8,8},
                {8,0,0,0,0,0,8},
                {8,0,0,0,0,0,8},
                {8,0,0,0,0,0,8},
                {8,0,0,0,0,0,8},
                {8,0,0,0,0,0,8},
                {8,8,8,8,8,8,8}
            };

            int mapSize = 7;

            // Here we will train a NN on the game to make good evaluations of the board
            // We will set 0 to be a Blue Sided and 1 to be Red Sided

            // We will use a field that is smaller than the nodes
            var nn = new NeuralNetWork(1, (mapSize-2)*(mapSize-2));

            // How many games we will run per run.
            int gamesPerSet = 200;
            int trainingSets = 100;

            double[,] trainingBoards;
            double[,] trainingValues;

            // We will get the lists of the gameboards and of the results for our NN to train on
            List<double[]> boards = new List<double[]>();
            List<double> results = new List<double>();


            // This is the main training loop that we will run
            while(trainingSets != 0)
            {
                // Clear all the boards before the run
                boards.Clear();
                results.Clear();

                // Run the Training Games per Set
                for(int c = 0; c < gamesPerSet; c++)
                {
                    // Create the Root Board every new game run
                    Board root = new Board(mapSize, -1, rootBoard, null, trainedModel, true);
                    int newBoards = 0;

                    // While we are not in an end state we run the while loop
                    while(!root.IsEndState(root))
                    {
                        // Create the Children of the Board
                        root.GenerateChildren();

                        if(root.children.Count == 0)
                        {
                            // This means there are no Children so we will leave the loop
                            break;
                        }

                        // Pick the one best according to the model
                        double bestMatrixScore = 0.5;
                        Board bestMove = root.children[0];

                        foreach(Board child in root.children)
                        {
                            var nnPredict = nn.Think(GetDoubleRepresentation(child.data));
                            if(root.currentMove == 1)
                            {
                                // We want to make the best move for BLUE Team so we are maximizing
                                if(bestMatrixScore < nnPredict[0,0])
                                {
                                    bestMatrixScore = nnPredict[0,0];
                                    bestMove = child;
                                }
                            }
                            else
                            {
                                // We want to get the best move for RED so we are minimizing
                                if(bestMatrixScore > nnPredict[0,0])
                                {
                                    bestMatrixScore = nnPredict[0,0];
                                    bestMove = child;
                                }
                            }
                        }  

                        // We found our best Child
                        boards.Add(GetSingleDouble(bestMove.data));
                        root = bestMove;
                        newBoards++;
                        root.EvaluateBoard();
                    }

                    // We want to make sure we know who Won
                    double reward = 0.5;
                    if(root.BlueScore > root.RedScore)
                    {
                        // Blue Won so we reward 0
                        reward = 0;
                    }
                    else if(root.BlueScore < root.RedScore)
                    {
                        reward = 1;
                    }

                    for(int vc = 0; vc < newBoards; vc++)
                    {
                        results.Add(reward);
                    }

                    newBoards = 0;
                }
                

                // We add things now to our training data
                trainingBoards = new double[boards.Count, (mapSize-2)*(mapSize-2)];
                trainingValues = new double[1,boards.Count];

                Console.WriteLine(boards.Count);

                for(int x = 0; x < boards.Count; x++)
                {
                    for(int y = 0; y < (mapSize-2)*(mapSize-2); y ++)
                    {
                        trainingBoards[x,y] = boards[x][y];
                    }

                    trainingValues[0,x] = results[x];
                }

                nn.Train(trainingBoards, NeuralNetWork.MatrixTranspose(trainingValues), 1000);
                Console.WriteLine("Trained On New Data");
                Console.WriteLine(trainingSets);
                Console.WriteLine(boards.Count);

                trainingSets--;
            }

            DisplayMatrix(nn.SynapsesMatrix);

            rnd = new Random();

            trainedModel = nn;

            // Main Function that will play the Game and Board

            PlayGame(rootBoard);
        }

        // We return the NN value of the current estimate
        public static double ReturnNNValue(Board gameBoard)
        {
            double[,] value = trainedModel.Think(GetDoubleRepresentation(gameBoard.data));
            return value[0,0];
        }

        // This is the main function that will take the input of the player and play MCTS against them on every move
        public static void PlayGame(int[,] rootBoard)
        {
            // We define how many games we will play each time that we run MCTS
            int iterationsForGames = 10000;

            // Define the map size for our player. +2 to map size to account for borders used in checkingA
            int mapSize = 7;

            // We  check if we have found the move that is optimal
            bool moveFound = false;

            // We create the original board and the Root state
            Board root = new Board(mapSize, -1, rootBoard, null, trainedModel, false);

            int moves = 1;

            // The main game loop that will run in order for you to play the computer
            while(!root.IsEndState(root))
            {
                moveFound = false;

                // We will start the game loop
                root.PrintSelfReadable();
                Console.WriteLine("RED TO MOVE COMPUTER IS THINKING... ");

                // This function will train the game loop for the given number of iterations
                TrainMCTS(iterationsForGames, root);

                root = root.ReturnBestChild();
                root.PrintSelfReadable();
                Console.WriteLine("NN Evaluation: " + ReturnNNValue(root));

                while(!moveFound)
                {
                    Console.WriteLine("BLUE TO MOVE. Please make your move: 'first x then y'  ");
                    var xx = Console.ReadLine();
                    if(xx == "reset")
                    {
                        PlayGame(rootBoard);
                        return;
                    }
                    int x = Int32.Parse(xx);
                    int y = Int32.Parse(Console.ReadLine());

                    foreach(Board child in root.children)

                    {
                        if(child.data[x,y] == -1)
                        {
                            root = child;
                            moveFound = true;
                            break;
                        }
                    }
                }

                moves++;
            }
        }


        // NOTE: CHANGE THESE SETTINGS IF YOU WANT TO CHANGE MAP SIZE
        public static double[,] GetDoubleRepresentation(int[,] boardData)
        {
            double[,] finalDouble = new double[1, (7-2)*(7-2)];

            for(int x = 1; x < boardData.GetLength(0) -1; x++)
            {
                for(int y = 1; y < boardData.GetLength(1) -1; y++)
                {
                    finalDouble[0, (5*(x-1)) + (y-1)] = boardData[x,y];
                }
            }

            return finalDouble;
        }

        // Functions to change formats to adapt to the NN
        public static double[] GetSingleDouble(int[,] boardData)
        {
            double[] finalDouble = new double[25];

            for(int x = 1; x < boardData.GetLength(0) -1; x++)
            {
                for(int y = 1; y < boardData.GetLength(1) -1; y++)
                {
                    finalDouble[(5*(x-1)) + (y-1)] = boardData[x,y];
                }
            }

            return finalDouble;
        }

        static void DisplayMatrix(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
        }

        // This Function that trains the MCTS given the number of iterations and the root that will be used as the origin of the MCTS
        public static void TrainMCTS(int iterations, Board root)
        {

            Board currentBoard = root;
            int timesSimulated = 0;

            // Used to calculate the time when needing to start a new iteration
            var watch = System.Diagnostics.Stopwatch.StartNew();

            // This is the for loop that will train the algorithm given the number of iterations that it needs to do
            for(int run = 0; run < iterations; run++)
            {
                currentBoard = root;

                // This is our single loop to run one MCTS iteration
                while(true)
                {
                    if(currentBoard.children.Count == 0)
                    {
                        // We have no children so we must create them
                        currentBoard.GenerateChildren();

                        if(currentBoard.children.Count == 0 || currentBoard.IsEndState(currentBoard))
                        {
                            currentBoard.PropogateSelf();
                            break;
                        }
                        else
                        {
                            // We have generated our children so now we can choose a random one at random and then make the rollout
                            int randomChild = rnd.Next(currentBoard.children.Count);
                            currentBoard.children[randomChild].MakeRollout();
                            break;
                        }
                    }

                    // We have children so we must take a look at and select the next best child
                    currentBoard = currentBoard.ReturnBestChild();
                }

                timesSimulated++;
                
                if(timesSimulated % 5000 == 0) 
                {
                    Console.WriteLine(timesSimulated);
                    watch.Stop();
                    Console.WriteLine("TIME TAKEN: " + watch.ElapsedMilliseconds);
                    watch.Restart();
                }
            }

            watch.Stop();
        }
    }
}
