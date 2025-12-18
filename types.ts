
export interface TrainingLog {
  episode: number;
  reward: number;
  loss: number;
  steps: number;
  timestamp: string;
}

export interface AgentState {
  x: number;
  y: number;
  goalX: number;
  goalY: number;
  obstacles: Array<{x: number, y: number}>;
  totalReward: number;
  step: number;
}

export enum ViewMode {
  DASHBOARD = 'DASHBOARD',
  SIMULATOR = 'SIMULATOR',
  CODE = 'CODE',
  COLAB = 'COLAB'
}
