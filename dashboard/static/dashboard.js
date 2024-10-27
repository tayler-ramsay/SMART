import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { AlertCircle, CheckCircle, Play, Pause, Square, Save } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from 'components/ui/card';
import { Alert, AlertDescription } from 'components/ui/alert';
import io from 'socket.io-client';

const socket = io('http://localhost:5002');

const LLMTrainingDashboard = () => {
  const [metrics, setMetrics] = useState({
    loss: [],
    accuracy: [],
    attentionEntropy: [],
    layerGradients: {},
    domainAccuracy: {},
    contextRetention: [],
    knowledgeConsistency: [],
    convergenceRate: 0,
    resourceUtilization: {
      gpu: 0,
      cpu: 0,
      memory: 0
    },
    alerts: []
  });

  const [trainingState, setTrainingState] = useState('running');
  const [selectedMetric, setSelectedMetric] = useState('loss');

  useEffect(() => {
    console.log("Requesting data from server");
    // Request data from the server
    socket.emit('request_data');

    // Listen for updates from the server
    socket.on('update_metrics', (data) => {
      console.log("Received metrics update", data);
      setMetrics(data);
    });

    return () => {
      console.log("Disconnecting from server");
      socket.off('update_metrics');
    };
  }, []);

  const handleTrainingControl = (action) => {
    console.log(`Sending training control action: ${action}`);
    setTrainingState(action === 'resume' ? 'running' : action);
    socket.emit('control_training', action);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto bg-gray-900 text-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">LLM Training Dashboard</h1>
          <p className="text-gray-400">Real-time training monitoring and analysis</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => handleTrainingControl(trainingState === 'running' ? 'pause' : 'resume')}
            className="flex items-center gap-2 px-4 py-2 rounded bg-blue-600 hover:bg-blue-700"
          >
            {trainingState === 'running' ? <Pause size={16} /> : <Play size={16} />}
            {trainingState === 'running' ? 'Pause' : 'Resume'}
          </button>
          <button 
            onClick={() => handleTrainingControl('stop')}
            className="flex items-center gap-2 px-4 py-2 rounded bg-red-600 hover:bg-red-700"
          >
            <Square size={16} />
            Stop
          </button>
          <button 
            onClick={() => handleTrainingControl('checkpoint')}
            className="flex items-center gap-2 px-4 py-2 rounded bg-green-600 hover:bg-green-700"
          >
            <Save size={16} />
            Checkpoint
          </button>
        </div>
      </div>

      {/* Main Metrics Chart */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Training Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <LineChart width={900} height={250} data={metrics[selectedMetric].map((value, index) => ({
              name: index,
              value
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#8884d8" />
            </LineChart>
          </div>
        </CardContent>
      </Card>

      {/* Resource Utilization */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>GPU Utilization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.resourceUtilization.gpu.toFixed(1)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full" 
                style={{width: `${metrics.resourceUtilization.gpu}%`}}
              ></div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>CPU Utilization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.resourceUtilization.cpu.toFixed(1)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-green-600 h-2.5 rounded-full" 
                style={{width: `${metrics.resourceUtilization.cpu}%`}}
              ></div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Memory Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.resourceUtilization.memory.toFixed(1)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-yellow-600 h-2.5 rounded-full" 
                style={{width: `${metrics.resourceUtilization.memory}%`}}
              ></div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Layer Gradients */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Layer-specific Gradients</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-6 gap-4">
            {Object.entries(metrics.layerGradients).map(([layer, value]) => (
              <div key={layer} className="text-center">
                <div className="text-sm text-gray-400">{layer}</div>
                <div className="text-lg font-bold">{value.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Domain Accuracy */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Domain-specific Accuracy</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            {Object.entries(metrics.domainAccuracy).map(([domain, accuracy]) => (
              <div key={domain} className="text-center">
                <div className="text-sm text-gray-400">{domain}</div>
                <div className="text-lg font-bold">{(accuracy * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      {metrics.alerts.length > 0 && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {metrics.alerts[metrics.alerts.length - 1]}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default LLMTrainingDashboard;
