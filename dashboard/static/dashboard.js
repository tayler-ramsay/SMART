// static/js/dashboard.js

// Initialize Socket.IO connection
const socket = io('http://localhost:5002');

// React Import
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { 
  AlertCircle, TrendingUp, Brain, Zap, 
  Play, Pause, Download, Share,
  Settings, Bell, AlertTriangle,
  Github, MessageSquare, Layers
} from 'lucide-react';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    loss: [],
    accuracy: [],
    validation_loss: [],
    perplexity: [],
    gradient_norm: [],
    learning_rate: [],
    epoch: 0,
    progress: 0,
    tokens_processed: 0,
    processing_speed: 0,
    gpu_utilization: 85,
    cpu_utilization: 45,
    memory_usage: 12.4,
    estimated_time: 3600,
    cost_per_hour: 2.5,
    stability_score: 0.85
  });

  const [trainingStatus, setTrainingStatus] = useState({
    isActive: true,
    lastCheckpoint: '2024-10-26 15:30:00',
    checkpointMetrics: { loss: 0.234, accuracy: 0.891 }
  });

  const [selectedTab, setSelectedTab] = useState('overview');

  // Socket.IO event handlers
  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to server');
    });

    socket.on('update_metrics', (newMetrics) => {
      setMetrics(prevMetrics => ({
        ...prevMetrics,
        ...newMetrics
      }));
    });

    // Request initial data
    socket.emit('request_data');

    // Set up periodic data requests
    const interval = setInterval(() => {
      socket.emit('request_data');
    }, 2000);

    return () => {
      clearInterval(interval);
      socket.off('update_metrics');
    };
  }, []);

  // Training control handlers
  const handleTrainingControl = (action) => {
    socket.emit('training_control', { action });
  };

  // Render components...
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">LLM Training Dashboard</h1>
          <div className="flex items-center mt-2 text-gray-500">
            <Github className="h-4 w-4 mr-2" />
            <a href="https://github.com/yourusername/llm-training-dashboard" 
               className="hover:text-blue-500 transition-colors">
              View on GitHub
            </a>
          </div>
        </div>
        <div className="flex gap-4">
          <Button 
            variant="outline"
            onClick={() => handleTrainingControl(trainingStatus.isActive ? 'pause' : 'resume')}
          >
            {trainingStatus.isActive ? 
              <Pause className="mr-2 h-4 w-4" /> : 
              <Play className="mr-2 h-4 w-4" />
            }
            {trainingStatus.isActive ? 'Pause Training' : 'Resume Training'}
          </Button>
          <Button 
            variant="outline"
            onClick={() => handleTrainingControl('checkpoint')}
          >
            <Download className="mr-2 h-4 w-4" />
            Save Checkpoint
          </Button>
        </div>
      </div>

      {/* Main Progress Card */}
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
          <CardDescription>Overall completion and key metrics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Progress value={metrics.progress} className="h-2" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Epoch:</span>
              <span className="ml-2 font-medium">{metrics.epoch} of 10</span>
            </div>
            <div>
              <span className="text-gray-500">Tokens:</span>
              <span className="ml-2 font-medium">
                {(metrics.tokens_processed / 1e6).toFixed(1)}M
              </span>
            </div>
            <div>
              <span className="text-gray-500">Speed:</span>
              <span className="ml-2 font-medium">
                {(metrics.processing_speed / 1000).toFixed(1)}K tokens/s
              </span>
            </div>
            <div>
              <span className="text-gray-500">GPU:</span>
              <span className="ml-2 font-medium">{metrics.gpu_utilization}%</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Training Loss</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics.loss.map((loss, i) => ({
                  epoch: i + 1,
                  training: loss,
                  validation: metrics.validation_loss[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="training" 
                    name="Training Loss" 
                    stroke="#ef4444" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="validation" 
                    name="Validation Loss" 
                    stroke="#3b82f6" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics.perplexity.map((ppl, i) => ({
                  epoch: i + 1,
                  perplexity: ppl,
                  accuracy: metrics.accuracy[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="perplexity" 
                    name="Perplexity" 
                    stroke="#22c55e" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    name="Accuracy" 
                    stroke="#eab308" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Mount React app
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<Dashboard />);