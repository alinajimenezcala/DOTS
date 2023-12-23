using System;

namespace VRAG
{
    // This is the class that will contain the board information. The information that will be included in the file is the following
    // data: 2D Vector that willa actually contain the board
    // parent: board that is the parent of this state
    // children: the children on the board that will be all the possible moves that can be made
    // Blue Score: the score of the blue player
    // Red Score: the score of the red player
    // visitations: how many times was this state visited
    // wins: how many games were won from this state

    class Board
    {
        // Data stored by this class
        public int[,] data;
        public int currentMove;
        int mapSize;

        public int RedScore;
        public int BlueScore;

        int redWins;
        int blueWins;
        int timesPlayed;

        public int smallestX;
        public int smallestY;
        public int largestX;
        public int largestY;

        bool canSkinEval;

        public int x;
        public int y;

        public List<Board> children = new List<Board>();
        public Board parent;

        static Random rnd;

        static NeuralNetWork trainedNN;

        bool training;

        // This is the Board Object that we will use to represent the individual game state
        public Board(int size, int move, int[,] parentData, Board parentBoard, NeuralNetWork nn, bool isTraining)
        {
            mapSize = size;
            currentMove = move;
            data = new int[mapSize,mapSize];
            data = parentData.Clone() as int[,];
            RedScore = 0;
            BlueScore = 0;
            parent = parentBoard;

            canSkinEval = false;
            training = isTraining;

            x = -1;
            y = -1;

            redWins = 0;
            blueWins = 0;
            timesPlayed = 0;

            trainedNN = nn;

            rnd = new Random();
        }

        // This is the code of the Function that prints the Board
        public void PrintSelf()
        {
            Console.WriteLine("BLUE SCORE: " + BlueScore);
            Console.WriteLine("RED SCORE: " + RedScore);
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    Console.Write(data[j,i] + "\t");
                }
                Console.WriteLine();
            }
        }

        // We will set the x and the y of the board. This will also dictate our working perimiter for the evaluation of the board
        public void SetMyXandY(int setX, int setY)
        {
            x = setX;
            y = setY;

            if(parent.x == -1)
            {
                largestX = setX + 1;
                largestY = setY + 1;

                smallestX = setX - 1;
                smallestY = setY - 1;

                canSkinEval = true;

                return;
            }

            // If our X is smaller than the parents X or larger than our parents X we will use our X. Otherwise we will use our parents;
            smallestX = Math.Min(parent.smallestX, x-1);
            smallestY = Math.Min(parent.smallestY, y-1);

            largestX = Math.Max(parent.largestX, x+1);
            largestY = Math.Max(parent.largestY, y+1);
        }


        // This function prints a lot of information about the board state that is not printed in other print function
        public void PrintSelfReadable()
        {

            Console.WriteLine("----------------------------");
            Console.WriteLine("BLUE SCORE: " + BlueScore);
            Console.WriteLine("RED SCORE: " + RedScore);
            Console.WriteLine("TIMES PLAYED: " + timesPlayed);
            Console.WriteLine("RED WINS " + redWins);
            Console.WriteLine("BLUE WINS " + blueWins);
            Console.WriteLine("TOP X AND Y " + largestX + " " + largestY);
            Console.WriteLine("BOTTOM X AND Y " + smallestX + " " + smallestY);

            for (int j = 0; j < data.GetLength(0); j++)
            {
                for (int i = 0; i < data.GetLength(1); i++)
                {
                    if(data[i,j] == 1)
                    {
                        Console.Write("X" + "\t");
                    }
                    else if(data[i,j] == 3)
                    {
                        Console.Write(".x." + "\t");
                    }
                    else if(data[i,j] == -1)
                    {
                        Console.Write("O" + "\t");
                    }
                    else if(data[i,j] == -3)
                    {
                        Console.Write(".o." + "\t");
                    }
                    else
                    {
                        if(i == 0 || i == data.GetLength(0)-1)
                        {
                            Console.Write(j + "\t");
                        }
                        else if(j == 0 || j == data.GetLength(0)-1)
                        {
                            Console.Write(i + "\t");
                        }
                        else
                        {
                            Console.Write("-" + "\t");
                        }
                    }
                }
                Console.WriteLine();
            }
        }

        // This function returns the best child out of the current children so that it can be used for our decision
        public Board ReturnBestChild()
        {
            // This will return the best Child to use form the list
            Board returnBoard;
            double bestValue = 0;

            returnBoard = children[0];
            bestValue = children[0].BestValue();

            // We select the best move for BLUE
            foreach(Board child in children)
            {
                double valueToCheck = child.BestValue();
                if(valueToCheck.CompareTo(bestValue) > 0)
                {
                    returnBoard = child;
                    bestValue = valueToCheck;
                }
            }

            return returnBoard;
        }

        // Generate all the Children for the board
        public void GenerateChildren()
        {
            // For every 0 we want to create and add a child to the children
            for (int i = 0; i < mapSize; i++)
            {
                for (int j = 0; j < mapSize; j++)
                {
                    if(data[i,j] == 0)
                    {
                        Board newBoard = new Board(mapSize, currentMove*-1, data, this, trainedNN, training);
                        newBoard.BlueScore = BlueScore;
                        newBoard.RedScore = RedScore;
                        newBoard.data[i,j] = currentMove*-1;

                        // We want to give the X and Y of the last move to the board
                        newBoard.SetMyXandY(i, j);

                        newBoard.EvaluateBoard();
                        children.Add(newBoard);
                    }
                }
            }
        }

        // This function back propogates through each prior board state and then sets its new MCTS score
        public void PropogateSelf()
        {
            if(RedScore > BlueScore)
            {
                BackPropogate(1);
            }
            else if(BlueScore > RedScore)
            {
                BackPropogate(-1);
            }
            else
            {
                BackPropogate(0);
            }
        }
        public static double[,] GetDoubleRepresentation(int[,] boardData)
        {
            double[,] finalDouble = new double[1, 25];

            for(int x = 1; x < boardData.GetLength(0) -1; x++)
            {
                for(int y = 1; y < boardData.GetLength(1) -1; y++)
                {
                    finalDouble[0, (5*(x-1)) + (y-1)] = boardData[x,y];
                }
            }

            return finalDouble;
        }

        // Get the value of the board. We will return this with consideration of the current move
        public double BestValue()
        {
            double winRate = 0;
            double c = Math.Sqrt(2);
            double exploration = 0;
            double scoreGap;

            // This will return the value of the board
            if(currentMove == 1)
            {
                // We will return with consideration of the Red Value

                if(timesPlayed == 0)
                {
                    return 9999999f;
                }

                scoreGap = (RedScore-BlueScore);

                // We will get a value win rate
                winRate = (double)redWins/(double)timesPlayed - (double)blueWins/(double)timesPlayed;

                exploration = (c * (double)Math.Sqrt(Math.Log(parent.timesPlayed)/(double)timesPlayed));

                // Console.WriteLine("WIN RATE " + winRate);

                if(training)
                {
                    return winRate + exploration + scoreGap + trainedNN.Think(GetDoubleRepresentation(data))[0,0];
                }
                else
                {
                    return winRate + exploration + scoreGap;
                }
            }
            else
            {
                // We will return with consideration of the Blue Value

                if(timesPlayed == 0)
                {
                    return 9999999f;
                }

                scoreGap = (BlueScore-RedScore);

                // We will get a value win rate
                winRate = (double)blueWins/(double)timesPlayed - (double)redWins/(double)timesPlayed;

                exploration = (c * (double)Math.Sqrt(Math.Log(parent.timesPlayed)/(double)timesPlayed));

                if(training)
                {
                    return winRate + exploration + scoreGap + (1-trainedNN.Think(GetDoubleRepresentation(data))[0,0]);
                }
                else
                {
                    return winRate + exploration + scoreGap;
                }
            }
        }

        // We print out all the Children 
        public void PrintAllChildren()
        {
            foreach(Board child in children)
            {
                Console.WriteLine("Value of the Child " + child.BestValue());
                child.PrintSelfReadable();
            }
        }

        // We print the best response possible after running all the MCTS simulations
        public void PrintBestResponse()
        {
            Board bestChild = children[0];
            double bestScore = children[0].BestValue();

            double valueToCheck = 0;

            foreach(Board child in children)
            {
                valueToCheck = child.BestValue();
                if(valueToCheck.CompareTo(bestScore) > 0)
                {
                    bestChild = child;
                    bestScore = valueToCheck;
                }
            }

            bestChild.PrintSelfReadable();
            Console.WriteLine(bestChild.BestValue());
        }

        // RollOut
        // This is the rollout function that will rollout from this board and will return the win or a loss depending on who is playing this match
        public void MakeRollout()
        {
            Board rolloutBoard = new Board(mapSize, currentMove, data, null, trainedNN, training);
            int randomIndex = 0;
            int loopBreak = 0;

            // We will rollout while the board is not in an end state
            while(!IsEndState(rolloutBoard))
            {
                if(loopBreak > 1000)
                {
                    Console.WriteLine("Too DEEEP! ");
                    break;
                }
                // If the number of Children is 0 then we will Generate them
                if(rolloutBoard.children.Count == 0)
                {
                    rolloutBoard.GenerateChildren();
                }
                else
                {
                    Console.WriteLine("Rollingout on a Board with Children! ");
                    return;
                }
                if(rolloutBoard.children.Count == 0)
                {
                    break;
                }

                // As we have not reached the end we will choose a random child and continue the rollout
                randomIndex = rnd.Next(rolloutBoard.children.Count);
                rolloutBoard = rolloutBoard.children[randomIndex];
                // rolloutBoard.PrintSelfReadable();
                loopBreak++;
            }

            // We have made our rollout and should now backpropogate the data
            if(rolloutBoard.BlueScore > rolloutBoard.RedScore)
            {
                // Blue won the game
                this.BackPropogate(-1);
            }
            else if(rolloutBoard.RedScore > rolloutBoard.BlueScore)
            {
                // Red won the game
                this.BackPropogate(1);
            }
            else
            {
                // no one has won the game
                this.BackPropogate(0);
            }

            // rolloutBoard.PrintSelfReadable();
            // rolloutBoard.PrintSelf();
        }

        // The actual back propogate function that will back propogate based on what who won the last game
        public void BackPropogate(int outcome)
        {
            Board parentBoard = this;
            while(parentBoard != null)
            {
                // We will increase the win rate of each board and then we will also add the games played
                parentBoard.timesPlayed++;

                if(outcome == -1)
                {
                    parentBoard.blueWins++;
                }
                else if(outcome == 1)
                {
                    parentBoard.redWins++;
                }

                parentBoard = parentBoard.parent;
            }
        }

        // Is this an end state
        public bool IsEndState(Board checkBoard)
        {
            if(checkBoard.RedScore > 10 || checkBoard.BlueScore > 10)
            {
                return true;
            }
            return false;
        }

        // We check the state of the board using a modified flood fill algorithm. Also some corner cutting is used here as well
        public void EvaluateBoard()
        {

            if(canSkinEval)
            {
                return;
            }

            // If there are less than 2 neightbours we will not do anything and inherit values
            int connections = 0;

            connections += CheckNeighbour(x-1,y);
            connections += CheckNeighbour(x+1,y);
            connections += CheckNeighbour(x,y-1);
            connections += CheckNeighbour(x,y+1);
            connections += CheckNeighbour(x-1,y-1);
            connections += CheckNeighbour(x+1,y+1);
            connections += CheckNeighbour(x-1,y+1);
            connections += CheckNeighbour(x+1,y-1);

            if(connections < 2)
            {
                // We want to take the values from the parent
                RedScore = parent.RedScore;
                BlueScore = parent.BlueScore;
                return;
            }

            if(currentMove == 1)
            {
                EvaluateBoardRed();
            }
            else
            {
                EvaluateBoardBlue();
            }
        }

        int CheckNeighbour(int xToCheck, int yToCheck)
        {

            if(xToCheck > 0 || yToCheck > 0 || xToCheck < mapSize || yToCheck < mapSize)
            {
                if(data[xToCheck,yToCheck] == currentMove)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }

            return 0;
        }

        // Evaluation Key
        // WALL: 8
        // EMPTY: 0
        // RED DOT: 1
        // BLUE DOT: -1
        // CAPTURED BY RED: 2
        // CAPTURED RED DOT: 3
        // CAPTURED BY BLUE: -2
        // CAPTURED BLUE DOT: -3

        // Evaluate RED
        // We run this when our last move was RED
        // RED EVAL KEY
        // 8->88
        // 0->7
        // 1->1
        // -1-> -11
        // 2->22
        // 3->33

        public void EvaluateBoardRed()
        {
            // We start by filling the board
            redFill(smallestX,smallestY);

            // After filling each 0 is a captured space by RED and each -1 is a Captured BLUE
            // We will go through each value and set it back and update all the scores
            for (int i = smallestX; i <= largestX; i++)
            {
                for (int j = smallestY; j <= largestY; j++)
                {
                    // We loop through each value
                    switch(data[i,j])
                    {
                        case 88:
                            data[i,j] = 8;
                            break;
                        case 7:
                            data[i,j] = 0;
                            break;
                        case -11:
                            data[i,j] = -1;
                            break;
                        case 22:
                            data[i,j] = 2;
                            break;
                        case 33:
                            data[i,j] = 3;
                            break;
                        case 0:
                            data[i,j] = -2;
                            break;
                        case -1:
                            data[i,j] = -3;
                            RedScore++;
                            break;
                    }
                }
            }

        }
        
        // This is the fill function for RED

        void redFill(int x, int y)
        {
            // If we are hitting new values then return
            if((x < smallestX) || (y < smallestY) || (x > largestX) || (y > largestY))
            {
                return;
            }

            int value = data[x,y];

            if((value == 88) || (value == 7) || (value == 1) || (value == -11) || (value == 22) || (value == 33) || (value == -33) || (value == -22))
            {
                return;
            }

            // We have a value that we can change so we run our switch statement
            switch(value)
            {
                case 0:
                    data[x,y] = 7;
                    break;
                case 8:
                    data[x,y] = 88;
                    break;
                case -1:
                    data[x,y] = -11;
                    break;
                case 2:
                    data[x,y] = 22;
                    break;
                case 3:
                    data[x,y] = 33;
                    break;
                case -2:
                    data[x,y] = -22;
                    break;
                case -3:
                    data[x,y] = -33;
                    break;
            }

            redFill(x+1,y);
            redFill(x-1,y);
            redFill(x,y+1);
            redFill(x,y-1);
            
        }

        // Evaluate BLUE
        // We run this when our last move was BLUE
        // BLUE EVAL KEY
        // 8->88
        // 0->7
        // 1->11
        // -1-> -1
        // -2->-22
        // -3->-33

        // Evaluate BLUE
        public void EvaluateBoardBlue()
        {
            // We start by filling the board
            blueFill(smallestX,smallestY);

            // After filling each 0 is a captured space by BLUE and each 1 is a Captured BLUE
            // We will go through each value and set it back and update all the scores
            for (int i = smallestX; i <= largestX; i++)
            {
                for (int j = smallestY; j <= largestY; j++)
                {
                    // We loop through each value
                    switch(data[i,j])
                    {
                        case 88:
                            data[i,j] = 8;
                            break;
                        case 7:
                            data[i,j] = 0;
                            break;
                        case 11:
                            data[i,j] = 1;
                            break;
                        case -22:
                            data[i,j] = -2;
                            break;
                        case -33:
                            data[i,j] = -3;
                            break;
                        case 0:
                            data[i,j] = 2;
                            break;
                        case 22:
                            data[i,j] = 2;
                            break;
                        case 33:
                            data[i,j] = 3;
                            break;
                        case 1:
                            data[i,j] = 3;
                            BlueScore++;
                            break;
                    }
                }
            }

        }

        void blueFill(int x, int y)
        {
            // If we are hitting new values then return
            if((x < smallestX) || (y < smallestY) || (x > largestX) || (y > largestY))
            {
                return;
            }

            int value = data[x,y];

            if((value == 88) || (value == 7) || (value == -1) || (value == 11) || (value == -22) || (value == -33) || (value == 33) || (value == 22))
            {
                return;
            }

            // We have a value that we can change so we run our switch statement
            switch(value)
            {
                case 0:
                    data[x,y] = 7;
                    break;
                case 8:
                    data[x,y] = 88;
                    break;
                case 1:
                    data[x,y] = 11;
                    break;
                case -2:
                    data[x,y] = -22;
                    break;
                case -3:
                    data[x,y] = -33;
                    break;
                case 2:
                    data[x,y] = 22;
                    break;
                case 3:
                    data[x,y] = 33;
                    break;
                default:
                    Console.WriteLine("We are hitting something " + data[x,y]);
                    break;
            }

            blueFill(x+1,y);
            blueFill(x-1,y);
            blueFill(x,y+1);
            blueFill(x,y-1);
        }
    }
}