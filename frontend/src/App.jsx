import React, { useState, useEffect, useRef } from 'react';
import { 
  LayoutDashboard, 
  Map, 
  Layers, 
  History, 
  BarChart3, 
  Cpu, 
  Upload, 
  Download, 
  Trash2, 
  RefreshCw, 
  FileText, 
  Search, 
  Filter, 
  AlertCircle, 
  CheckCircle2, 
  HelpCircle,
  ArrowRight,
  TrendingUp,
  FileSpreadsheet,
  AlertTriangle,
  Check
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Pie, Line, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = import.meta.env.VITE_API_BASE || (window.location.origin.includes('localhost:5173') ? 'http://localhost:5000' : window.location.origin);

const CLASSES = [
  "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
  "Industrial", "Pasture", "PermanentCrop", "Residential", 
  "River", "SeaLake"
];

const CLASS_COLORS = {
  AnnualCrop: "#86efac",
  Forest: "#15803d",
  HerbaceousVegetation: "#4ade80",
  Highway: "#64748b",
  Industrial: "#a21caf",
  Pasture: "#bef264",
  PermanentCrop: "#22c55e",
  Residential: "#f97316",
  River: "#3b82f6",
  SeaLake: "#1e3a8a"
};

const CLASS_DESC = {
  AnnualCrop: "Plantations harvested annually (e.g. wheat, soybeans)",
  Forest: "Dense tree canopy and woodland reserves",
  HerbaceousVegetation: "Natural pastures, shrubs, and wild grass",
  Highway: "Transportation routes, asphalt, and concrete roads",
  Industrial: "Factories, warehouses, and structural complexes",
  Pasture: "Managed grazing fields and meadows",
  PermanentCrop: "Long-term orchards, vineyards, and plantations",
  Residential: "Residential housing, towns, and suburban tracts",
  River: "Flowing freshwater courses and streams",
  SeaLake: "Still water bodies, lakes, seas, and ocean ports"
};

function App() {
  const singleInputRef = useRef(null);
  const batchInputRef = useRef(null);
  const oldInputRef = useRef(null);
  const newInputRef = useRef(null);

  const [activeTab, setActiveTab] = useState('dashboard');
  const [history, setHistory] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [training, setTraining] = useState(false);
  const [evaluatingExt, setEvaluatingExt] = useState(false);
  const [message, setMessage] = useState(null);

  // Single Predict Page State
  const [singleImage, setSingleImage] = useState(null);
  const [singlePreview, setSinglePreview] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [analyzingSingle, setAnalyzingSingle] = useState(false);

  // Batch Predict Page State
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [analyzingBatch, setAnalyzingBatch] = useState(false);

  // Change Detection Page State
  const [oldImage, setOldImage] = useState(null);
  const [oldPreview, setOldPreview] = useState(null);
  const [newImage, setNewImage] = useState(null);
  const [newPreview, setNewPreview] = useState(null);
  const [changeResult, setChangeResult] = useState(null);
  const [detectingChange, setDetectingChange] = useState(false);

  // Grid Classifier State
  const [gridImage, setGridImage] = useState(null);
  const [gridPreview, setGridPreview] = useState(null);
  const [gridResult, setGridResult] = useState(null);
  const [analyzingGrid, setAnalyzingGrid] = useState(false);
  const [hoveredGridCell, setHoveredGridCell] = useState(null);

  // History Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [classFilter, setClassFilter] = useState('');

  useEffect(() => {
    fetchHistory();
    fetchAnalytics();
  }, []);

  const handleGridUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setGridImage(file);
      setGridPreview(URL.createObjectURL(file));
      setGridResult(null);
    }
  };

  const triggerGridPredict = async () => {
    if (!gridImage) return;
    setAnalyzingGrid(true);
    const formData = new FormData();
    formData.append("image", gridImage);

    try {
      const res = await fetch(`${API_BASE}/grid_predict`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.predictions) {
        setGridResult(data);
        showNotification("Multi-scale grid slicing classification completed!", "success");
      } else {
        showNotification(data.message || "Failed to classify grid", "error");
      }
    } catch (err) {
      showNotification("Server error during grid classification", "error");
    } finally {
      setAnalyzingGrid(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setHistory(data);
      }
    } catch (err) {
      console.error("Error fetching history:", err);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_BASE}/analytics`);
      const data = await res.json();
      if (data && !data.status) {
        setAnalytics(data);
      }
    } catch (err) {
      console.error("Error fetching analytics:", err);
    }
  };

  const triggerEvalExternal = async () => {
    setEvaluatingExt(true);
    showNotification("Evaluating model on external validation images...", "info");
    try {
      const res = await fetch(`${API_BASE}/eval_external`, { method: "POST" });
      const data = await res.json();
      if (data.status === "success" || data.external_accuracy) {
        showNotification(`External Evaluation Complete! Accuracy: ${data.external_accuracy}`, "success");
        fetchAnalytics();
      } else {
        showNotification(data.message || "External evaluation notice", "info");
      }
    } catch (err) {
      showNotification("Error connecting to external evaluation endpoint", "error");
    } finally {
      setEvaluatingExt(false);
    }
  };

  const showNotification = (text, type = "success") => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 5000);
  };

  const handleTrain = async (mode = "demo") => {
    const isProduction = mode === "production";
    const confirmMsg = isProduction 
      ? "Are you sure you want to train the PRODUCTION model? This will run full training and update the primary model file (resnet50v2_eurosat.keras)." 
      : "Start Quick Trial training? (This will train a demo model without overwriting the production model).";
    
    if (!confirm(confirmMsg)) return;

    setTraining(true);
    showNotification(`Initiated ${isProduction ? 'PRODUCTION' : 'DEMO'} model training. Please wait...`, "info");
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, quick: !isProduction })
      });
      const data = await res.json();
      if (data.status === "success") {
        const acc = data.metrics?.accuracy || "Complete";
        showNotification(`${isProduction ? 'Production' : 'Demo'} training completed! Test Accuracy: ${acc}`, "success");
        fetchAnalytics();
      } else {
        showNotification(`Training failed: ${data.message}`, "error");
      }
    } catch (err) {
      showNotification("Error connecting to training endpoint", "error");
    } finally {
      setTraining(false);
    }
  };

  const handleSingleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSingleImage(file);
      setSinglePreview(URL.createObjectURL(file));
      setSingleResult(null);
    }
  };

  const triggerSinglePredict = async () => {
    if (!singleImage) return;
    setAnalyzingSingle(true);
    const formData = new FormData();
    formData.append("image", singleImage);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.prediction) {
        setSingleResult(data);
        showNotification(`Image classified as ${data.prediction}!`, "success");
        fetchHistory();
        fetchAnalytics();
      } else {
        showNotification(data.message || "Failed to classify image", "error");
      }
    } catch (err) {
      showNotification("Server error during single classification", "error");
    } finally {
      setAnalyzingSingle(false);
    }
  };

  const handleBatchUpload = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setBatchFiles(files);
      setBatchResults([]);
    }
  };

  const triggerBatchPredict = async () => {
    if (batchFiles.length === 0) return;
    setAnalyzingBatch(true);
    const formData = new FormData();
    batchFiles.forEach(file => {
      formData.append("images", file);
    });

    try {
      const res = await fetch(`${API_BASE}/batch_predict`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "success") {
        setBatchResults(data.predictions);
        showNotification(`Successfully processed ${data.predictions.length} images!`, "success");
        fetchHistory();
        fetchAnalytics();
      } else {
        showNotification(data.message || "Batch prediction failed", "error");
      }
    } catch (err) {
      showNotification("Server connection error during batch classification", "error");
    } finally {
      setAnalyzingBatch(false);
    }
  };

  const triggerChangeDetection = async () => {
    if (!oldImage || !newImage) return;
    setDetectingChange(true);
    const formData = new FormData();
    formData.append("old_image", oldImage);
    formData.append("new_image", newImage);

    try {
      const res = await fetch(`${API_BASE}/change_detection`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.impact) {
        setChangeResult(data);
        showNotification("Change detection analysis complete!", "success");
        fetchHistory();
        fetchAnalytics();
      } else {
        showNotification(data.message || "Failed to analyze changes", "error");
      }
    } catch (err) {
      showNotification("Server connection error during change detection", "error");
    } finally {
      setDetectingChange(false);
    }
  };

  const deleteHistoryRecord = async (id) => {
    if (!confirm("Are you sure you want to delete this prediction record?")) return;
    try {
      const res = await fetch(`${API_BASE}/history/${id}`, {
        method: "DELETE"
      });
      const data = await res.json();
      if (data.status === "success") {
        showNotification("Record deleted successfully", "success");
        fetchHistory();
        fetchAnalytics();
      }
    } catch (err) {
      showNotification("Failed to delete record", "error");
    }
  };

  const exportBatchToCSV = () => {
    if (batchResults.length === 0) return;
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Image Name,Prediction,Confidence (%),Uncertainty Flag\n";
    batchResults.forEach(r => {
      csvContent += `"${r.image_name}","${r.prediction}",${r.confidence},"${r.is_low_confidence ? 'Uncertain' : 'High'}"\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "batch_predictions_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredHistory = history.filter(item => {
    const matchesSearch = item.image_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          item.prediction.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesClass = classFilter === '' || item.prediction === classFilter;
    return matchesSearch && matchesClass;
  });

  const totalAnalyzed = history.length;
  const avgConf = history.length > 0 ? (history.reduce((acc, h) => acc + h.confidence, 0) / history.length).toFixed(1) : "0.0";
  
  const modelMetrics = analytics?.model_metrics;
  const prodAccuracy = modelMetrics?.evaluated ? modelMetrics.accuracy : "Not Evaluated";

  const extMetrics = analytics?.external_metrics;
  const extAccuracy = extMetrics?.status === "success" ? extMetrics.external_accuracy : "No Data";

  const getPieChartData = () => {
    if (!analytics || !analytics.class_distribution) return { labels: [], datasets: [] };
    const labels = Object.keys(analytics.class_distribution);
    const data = Object.values(analytics.class_distribution);
    const backgroundColors = labels.map(l => CLASS_COLORS[l] || "#cbd5e1");
    
    return {
      labels,
      datasets: [{
        data,
        backgroundColor: backgroundColors,
        borderWidth: 1,
      }]
    };
  };

  const getLineChartData = () => {
    if (!analytics || !analytics.trends) return { labels: [], datasets: [] };
    const labels = Object.keys(analytics.trends);
    const data = Object.values(analytics.trends);
    
    return {
      labels,
      datasets: [{
        label: "Predictions Logged",
        data,
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        tension: 0.3,
        fill: true,
      }]
    };
  };

  const getBarChartData = () => {
    if (!analytics || !analytics.model_metrics?.history) return { labels: [], datasets: [] };
    const hist = analytics.model_metrics.history;
    const labels = hist.accuracy.map((_, i) => `Epoch ${i + 1}`);
    
    return {
      labels,
      datasets: [
        {
          label: "Training Accuracy",
          data: hist.accuracy,
          backgroundColor: "rgba(59, 130, 246, 0.8)",
          borderColor: "#3b82f6",
          borderWidth: 1
        },
        {
          label: "Validation Accuracy",
          data: hist.val_accuracy,
          backgroundColor: "rgba(34, 197, 94, 0.8)",
          borderColor: "#22c55e",
          borderWidth: 1
        }
      ]
    };
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-800 font-sans overflow-hidden">
      
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-white text-slate-800 flex flex-col flex-shrink-0 border-r border-slate-200 shadow-sm z-10">
        <div className="p-6 border-b border-slate-200 flex items-center gap-3">
          <Map className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-base font-bold leading-tight font-display tracking-tight text-slate-900">GIS Satellite</h1>
            <p className="text-xs text-blue-600 font-medium">Land Cover System</p>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
            { id: 'classify', label: 'Land Classification', icon: Cpu },
            { id: 'grid', label: 'Grid Classifier', icon: Layers },
            { id: 'batch', label: 'Batch Processing', icon: Layers },
            { id: 'change', label: 'Change Detection', icon: RefreshCw },
            { id: 'history', label: 'Prediction Logs', icon: History },
            { id: 'analytics', label: 'Analytics & Validation', icon: BarChart3 },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab.id 
                  ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-sm' 
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <tab.icon className={`w-5 h-5 flex-shrink-0 ${activeTab === tab.id ? 'text-blue-600' : 'text-slate-400'}`} />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-200 bg-slate-50 text-center">
          <p className="text-xs text-slate-400 font-mono">MCA Project v2.0</p>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 z-10">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold tracking-tight text-slate-800 font-display">
              {activeTab === 'dashboard' && 'Dashboard Overview'}
              {activeTab === 'classify' && 'Satellite Classification & Grad-CAM'}
              {activeTab === 'grid' && 'Overlapping Multi-Scale Grid Classifier'}
              {activeTab === 'batch' && 'Batch Image Processing'}
              {activeTab === 'change' && 'GIS Temporal Change Detection'}
              {activeTab === 'history' && 'Classification Records'}
              {activeTab === 'analytics' && 'EuroSAT & External Validation Analytics'}
            </h2>
          </div>

          <div className="flex items-center gap-3">
            {training && (
              <span className="flex items-center gap-2 text-xs font-semibold px-3 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-full animate-bounce">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-600" />
                Training Model...
              </span>
            )}
            <div className="h-8 w-px bg-slate-200"></div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-blue-100 border border-blue-200 flex items-center justify-center text-blue-700 font-bold text-sm">
                GI
              </div>
              <span className="text-sm font-semibold text-slate-700">GIS Admin</span>
            </div>
          </div>
        </header>

        {/* Scrolling View Container */}
        <div className="flex-1 overflow-y-auto p-8 relative">
          
          {/* Global Alert Notification */}
          {message && (
            <div className={`fixed top-20 right-8 z-50 flex items-center gap-3 px-5 py-4 rounded-xl shadow-xl border animate-slide-in ${
              message.type === 'error' ? 'bg-red-50 text-red-800 border-red-200' :
              message.type === 'info' ? 'bg-blue-50 text-blue-800 border-blue-200' :
              'bg-emerald-50 text-emerald-800 border-emerald-200'
            }`}>
              {message.type === 'error' ? <AlertCircle className="w-5 h-5 text-red-600" /> : <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
              <p className="text-sm font-medium">{message.text}</p>
            </div>
          )}

          {/* TAB 1: DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-8 animate-fade-in">
              
              {/* KPI Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                  { label: "Total Classified Images", value: totalAnalyzed, desc: "Logged prediction records", icon: Layers, color: "text-blue-600 bg-blue-50 border-blue-100" },
                  { label: "Production Benchmark Accuracy", value: prodAccuracy, desc: "EuroSAT test split performance", icon: Cpu, color: "text-emerald-600 bg-emerald-50 border-emerald-100" },
                  { label: "External Validation Accuracy", value: extAccuracy, desc: "Out-of-distribution test set", icon: Map, color: "text-purple-600 bg-purple-50 border-purple-100" },
                  { label: "Average Class Confidence", value: `${avgConf}%`, desc: "Inference probability mean", icon: TrendingUp, color: "text-amber-600 bg-amber-50 border-amber-100" }
                ].map((kpi, idx) => (
                  <div key={idx} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between hover:shadow-md transition-shadow">
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{kpi.label}</p>
                      <h3 className="text-2xl font-extrabold text-slate-900 tracking-tight font-display">{kpi.value}</h3>
                      <p className="text-[11px] text-slate-400">{kpi.desc}</p>
                    </div>
                    <div className={`p-3.5 rounded-xl border ${kpi.color}`}>
                      <kpi.icon className="w-6 h-6" />
                    </div>
                  </div>
                ))}
              </div>

              {/* Training and Information Panel */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Training Actions Card */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm lg:col-span-2 space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                    <div>
                      <h3 className="text-lg font-bold text-slate-800 font-display">Model Training Controller</h3>
                      <p className="text-xs text-slate-500">Train or fine-tune ResNet50V2 on EuroSAT with Data Augmentation</p>
                    </div>
                    <Cpu className="w-6 h-6 text-blue-500" />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    
                    {/* Trial Demo Mode */}
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 flex flex-col justify-between">
                      <div>
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-[10px] font-bold rounded uppercase">Trial Trial Mode</span>
                        <h4 className="text-sm font-bold text-slate-800 mt-1">Quick Demo Training</h4>
                        <p className="text-xs text-slate-500 leading-relaxed mt-1">
                          Trains a demo trial model on small subset. Saves to <code className="text-[10px] bg-slate-200 px-1 rounded">demo_model.keras</code> without overwriting production model.
                        </p>
                      </div>
                      <button
                        onClick={() => handleTrain('demo')}
                        disabled={training}
                        className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${training ? 'animate-spin' : ''}`} />
                        Run Demo Trial Training
                      </button>
                    </div>

                    {/* Production Model Training */}
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 flex flex-col justify-between">
                      <div>
                        <span className="px-2 py-0.5 bg-emerald-100 text-emerald-800 text-[10px] font-bold rounded uppercase">Production Update</span>
                        <h4 className="text-sm font-bold text-slate-800 mt-1">Train Production Model</h4>
                        <p className="text-xs text-slate-500 leading-relaxed mt-1">
                          Runs 2-stage transfer learning fine-tuning on full EuroSAT splits. Updates production weights file <code className="text-[10px] bg-slate-200 px-1 rounded">resnet50v2_eurosat.keras</code>.
                        </p>
                      </div>
                      <button
                        onClick={() => handleTrain('production')}
                        disabled={training}
                        className="w-full mt-4 bg-emerald-700 hover:bg-emerald-800 text-white font-semibold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        Train Production Model
                      </button>
                    </div>

                  </div>

                  <div className="bg-amber-50 border border-amber-200 rounded-xl p-3.5 text-xs text-amber-800 flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                    <div className="leading-relaxed">
                      <strong className="font-semibold text-amber-900">Environment Requirement:</strong> Model training requires the 2GB EuroSAT dataset installed in local development environments. In cloud production deployments, training is disabled and the web application uses the pre-trained ResNet50V2 model.
                    </div>
                  </div>
                </div>

                {/* EuroSAT Classes Checklist Card */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                  <h3 className="text-lg font-bold text-slate-800 font-display">Supported Categories</h3>
                  <div className="max-h-[320px] overflow-y-auto space-y-2 pr-1">
                    {CLASSES.map((cls) => (
                      <div key={cls} className="flex items-center justify-between p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLASS_COLORS[cls] }}></span>
                          <span className="font-semibold text-slate-700">{cls}</span>
                        </div>
                        <span className="text-[10px] text-slate-400 max-w-[140px] truncate text-right">
                          {cls === "AnnualCrop" || cls === "PermanentCrop" ? "Crop Category" : "Natural/Built"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

              </div>

              {/* History Preview Table */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800 font-display">Recent Classification Activity</h3>
                    <p className="text-xs text-slate-500">Most recent predictions generated by the system</p>
                  </div>
                  <button 
                    onClick={() => setActiveTab('history')}
                    className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1"
                  >
                    View All Logs
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-400 font-semibold bg-slate-50/50">
                        <th className="py-3 px-4">Preview</th>
                        <th className="py-3 px-4">Filename</th>
                        <th className="py-3 px-4">Classification</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {history.slice(0, 5).map((row) => (
                        <tr key={row.id} className="hover:bg-slate-50/50 transition-colors">
                          <td className="py-3 px-4">
                            <img 
                              src={`${API_BASE}${row.image_url}`} 
                              alt={row.prediction}
                              className="w-10 h-10 object-cover rounded-md border border-slate-200"
                              onError={(e) => { e.target.src = 'https://placehold.co/100x100?text=Satellite'; }}
                            />
                          </td>
                          <td className="py-3 px-4 font-semibold text-slate-700">{row.image_name}</td>
                          <td className="py-3 px-4">
                            <span className="px-2.5 py-1 rounded-full text-[10px] font-bold" style={{ backgroundColor: `${CLASS_COLORS[row.prediction]}20`, color: CLASS_COLORS[row.prediction] }}>
                              {row.prediction}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-mono font-semibold text-blue-600">{row.confidence.toFixed(1)}%</td>
                          <td className="py-3 px-4 text-slate-500">{new Date(row.timestamp).toLocaleString()}</td>
                        </tr>
                      ))}
                      {history.length === 0 && (
                        <tr>
                          <td colSpan="5" className="py-8 text-center text-slate-400 font-semibold bg-slate-50/20">
                            No classifications logged yet. Upload an image to start!
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          )}

          {/* TAB 2: LAND CLASSIFICATION (SINGLE PREDICT) */}
          {activeTab === 'classify' && (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-8 animate-fade-in">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Upload & Preview Card */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800 font-display">Satellite Image Upload</h3>
                    <p className="text-xs text-slate-500">Upload a single 64x64 EuroSAT patch or a high-resolution outside image</p>
                  </div>

                  <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative group min-h-[300px]">
                    <input 
                      type="file" 
                      accept="image/*"
                      ref={singleInputRef}
                      onChange={handleSingleUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    
                    {singlePreview ? (
                      <div className="w-full flex flex-col items-center justify-center relative">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSingleImage(null);
                            setSinglePreview(null);
                            setSingleResult(null);
                            if (singleInputRef.current) singleInputRef.current.value = "";
                          }}
                          className="absolute -top-4 -right-4 p-2 bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 rounded-full shadow-md z-20 transition-all"
                          title="Remove Image"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                        
                        <img 
                          src={singlePreview} 
                          alt="Satellite Preview" 
                          className="max-h-[240px] rounded-xl object-contain border border-slate-200 shadow-md"
                        />
                        {analyzingSingle && (
                          <div className="absolute inset-0 bg-blue-500/10 flex items-center justify-center overflow-hidden rounded-xl">
                            <div className="w-full h-1 bg-blue-500 shadow-lg shadow-blue-500/50 absolute animate-[scan_2s_infinite]"></div>
                          </div>
                        )}
                        <p className="text-xs text-slate-500 mt-4 font-mono font-semibold">{singleImage?.name}</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="p-4 bg-blue-50 border border-blue-100 rounded-2xl text-blue-600 inline-block">
                          <Upload className="w-8 h-8 mx-auto" />
                        </div>
                        <div>
                          <p className="text-sm font-bold text-slate-700">Drag & drop your satellite image here</p>
                          <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, or TIFF formats</p>
                        </div>
                        <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg text-xs transition-colors pointer-events-none">
                          Browse Local Files
                        </button>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={triggerSinglePredict}
                    disabled={!singleImage || analyzingSingle}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition-all shadow-md flex items-center justify-center gap-2 disabled:opacity-50 text-sm"
                  >
                    {analyzingSingle ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Running Multi-Crop TTA Inference...
                      </>
                    ) : (
                      <>
                        <Cpu className="w-4 h-4" />
                        Perform Land Classification
                      </>
                    )}
                  </button>
                </div>

                {/* Analysis Results Display */}
                <div className="border border-slate-200 rounded-2xl bg-slate-50/50 p-6 flex flex-col justify-between min-h-[400px]">
                  {singleResult ? (
                    <div className="space-y-6 flex-1 flex flex-col justify-between">
                      
                      {/* Classification Title & Calibrated Confidence */}
                      <div className="flex justify-between items-start">
                        <div className="space-y-1">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Predicted Land Cover</span>
                          <h4 className="text-3xl font-extrabold text-slate-900 font-display flex items-center gap-2">
                            {singleResult.prediction}
                            <span 
                              className="w-4 h-4 rounded-full border border-white shadow-sm"
                              style={{ backgroundColor: CLASS_COLORS[singleResult.prediction] }}
                            ></span>
                          </h4>
                          <p className="text-xs text-slate-500 leading-relaxed font-medium">
                            {CLASS_DESC[singleResult.prediction]}
                          </p>
                        </div>
                        
                        <div className="text-right space-y-1">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Confidence</span>
                          <h4 className="text-3xl font-extrabold text-blue-600 font-mono tracking-tight">
                            {singleResult.confidence.toFixed(1)}%
                          </h4>
                          
                          {/* Low Confidence Warning Badge */}
                          {singleResult.is_low_confidence && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                              <AlertTriangle className="w-3 h-3 text-amber-600" />
                              Uncertain / Low Confidence
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Grad-CAM Visualizer */}
                      <div className="space-y-2">
                        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Explainability Saliency (Grad-CAM)</h5>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <div className="border border-slate-200 rounded-lg bg-white p-2">
                              <img 
                                src={`${API_BASE}${singleResult.image_url}`} 
                                alt="Original" 
                                className="w-full aspect-square object-cover rounded-md"
                              />
                            </div>
                            <p className="text-[10px] text-center font-bold text-slate-500">Input RGB Image</p>
                          </div>
                          <div className="space-y-1">
                            <div className="border border-slate-200 rounded-lg bg-white p-2">
                              {singleResult.heatmap_url ? (
                                <img 
                                  src={`${API_BASE}${singleResult.heatmap_url}`} 
                                  alt="Heatmap" 
                                  className="w-full aspect-square object-cover rounded-md"
                                />
                              ) : (
                                <div className="w-full aspect-square bg-slate-100 flex items-center justify-center text-xs text-slate-400 p-2 text-center">
                                  Grad-CAM unavailable for this input format
                                </div>
                              )}
                            </div>
                            <p className="text-[10px] text-center font-bold text-slate-500 text-blue-600">Model Attention Heatmap</p>
                          </div>
                        </div>
                      </div>

                      {/* Top Predictions */}
                      <div className="space-y-3 bg-white p-4 rounded-xl border border-slate-200">
                        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Top Class Candidates</h5>
                        <div className="space-y-2">
                          {singleResult.top_predictions.slice(0, 3).map((item, idx) => (
                            <div key={idx} className="space-y-1">
                              <div className="flex justify-between text-xs font-semibold">
                                <span className="text-slate-700">{item.class}</span>
                                <span className="text-slate-500 font-mono">{item.confidence.toFixed(1)}%</span>
                              </div>
                              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div 
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{ 
                                    width: `${item.confidence}%`,
                                    backgroundColor: CLASS_COLORS[item.class] || '#3b82f6' 
                                  }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Download PDF Action */}
                      {singleResult.id && (
                        <a
                          href={`${API_BASE}/download_report/${singleResult.id}`}
                          className="w-full bg-slate-800 hover:bg-slate-900 text-white font-bold py-3.5 rounded-xl text-xs transition-colors flex items-center justify-center gap-2"
                        >
                          <FileText className="w-4 h-4" />
                          Download PDF Summary Report
                        </a>
                      )}
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4">
                      <div className="w-16 h-16 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400">
                        <HelpCircle className="w-8 h-8" />
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-slate-700 font-display">Awaiting Classification Input</h4>
                        <p className="text-xs text-slate-400 mt-1 max-w-[280px] mx-auto">
                          Upload an image and run classification to view land cover predictions, confidence scores, and Grad-CAM saliency maps.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* TAB 3: BATCH PROCESSING */}
          {activeTab === 'batch' && (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-8 animate-fade-in">
              <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                <div>
                  <h3 className="text-lg font-bold text-slate-800 font-display">Batch Image Processing</h3>
                  <p className="text-xs text-slate-500">Upload multiple satellite images concurrently to classify entire file sets</p>
                </div>
                <div className="flex gap-2">
                  {batchResults.length > 0 && (
                    <button
                      onClick={exportBatchToCSV}
                      className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-4 py-2 rounded-lg text-xs transition-colors flex items-center gap-2"
                    >
                      <FileSpreadsheet className="w-4 h-4" />
                      Export Results CSV
                    </button>
                  )}
                </div>
              </div>

              {/* Upload Multi Dropzone */}
              <div 
                onClick={() => batchInputRef.current && batchInputRef.current.click()}
                className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative min-h-[160px]"
              >
                <input 
                  type="file" 
                  accept="image/*"
                  multiple={true}
                  ref={batchInputRef}
                  onChange={handleBatchUpload}
                  onClick={(e) => e.stopPropagation()}
                  className="hidden"
                />
                <div className="space-y-2">
                  <div className="p-3 bg-blue-50 border border-blue-100 rounded-full text-blue-600 inline-block">
                    <Upload className="w-6 h-6 mx-auto" />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-slate-700">Select Multiple Satellite Images</p>
                    <p className="text-xs text-slate-400 mt-1">
                      {batchFiles.length > 0 ? `${batchFiles.length} files selected` : "Supports batch upload of JPG/PNG/TIFF files"}
                    </p>
                  </div>
                  {batchFiles.length > 0 && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setBatchFiles([]);
                        setBatchResults([]);
                        if (batchInputRef.current) batchInputRef.current.value = "";
                      }}
                      className="mt-2 bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 font-semibold px-3 py-1 rounded text-[10px] transition-all relative z-10"
                    >
                      Clear Files
                    </button>
                  )}
                </div>
              </div>

              {batchFiles.length > 0 && (
                <button
                  onClick={triggerBatchPredict}
                  disabled={analyzingBatch}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 text-xs"
                >
                  {analyzingBatch ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Processing {batchFiles.length} Images...
                    </>
                  ) : (
                    <>
                      <Layers className="w-4 h-4" />
                      Analyze Selected Batch Files
                    </>
                  )}
                </button>
              )}

              {batchResults.length > 0 && (
                <div className="border border-slate-200 rounded-xl overflow-hidden mt-6 shadow-sm">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-400 font-semibold bg-slate-50">
                        <th className="py-3 px-4">Thumbnail</th>
                        <th className="py-3 px-4">Filename</th>
                        <th className="py-3 px-4">Classification</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Uncertainty Flag</th>
                        <th className="py-3 px-4 text-center">Report</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {batchResults.map((r, index) => (
                        <tr key={index} className="hover:bg-slate-50/50 transition-colors">
                          <td className="py-3 px-4">
                            {r.image_url ? (
                              <img 
                                src={`${API_BASE}${r.image_url}`} 
                                alt={r.prediction}
                                className="w-10 h-10 object-cover rounded-md border border-slate-200"
                              />
                            ) : (
                              <div className="w-10 h-10 bg-slate-100 rounded-md border border-slate-200 flex items-center justify-center text-[8px] font-bold text-red-500">
                                Fail
                              </div>
                            )}
                          </td>
                          <td className="py-3 px-4 font-semibold text-slate-700">{r.image_name}</td>
                          <td className="py-3 px-4">
                            {r.status === "error" ? (
                              <span className="text-red-500 font-bold">Failed</span>
                            ) : (
                              <span className="px-2.5 py-1 rounded-full text-[10px] font-bold" style={{ backgroundColor: `${CLASS_COLORS[r.prediction]}20`, color: CLASS_COLORS[r.prediction] }}>
                                {r.prediction}
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4 font-mono font-semibold text-blue-600">
                            {r.status === "error" ? "-" : `${r.confidence.toFixed(1)}%`}
                          </td>
                          <td className="py-3 px-4">
                            {r.is_low_confidence ? (
                              <span className="px-2 py-0.5 text-[10px] font-bold bg-amber-100 text-amber-800 rounded border border-amber-200 flex items-center gap-1 w-max">
                                <AlertTriangle className="w-3 h-3 text-amber-600" /> Uncertain
                              </span>
                            ) : (
                              <span className="text-[10px] text-slate-400 font-medium">Normal</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-center">
                            {r.id && (
                              <a
                                href={`${API_BASE}/download_report/${r.id}`}
                                className="text-slate-500 hover:text-blue-600 inline-flex items-center gap-1 font-semibold"
                              >
                                <Download className="w-3.5 h-3.5" />
                                PDF
                              </a>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

            </div>
          )}

          {/* TAB 3.1: GRID CLASSIFIER */}
          {activeTab === 'grid' && (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-8 animate-fade-in">
              <div className="border-b border-slate-100 pb-4">
                <h3 className="text-lg font-bold text-slate-800 font-display">Overlapping Multi-Scale Grid Classifier</h3>
                <p className="text-xs text-slate-500">Upload a large satellite composite. The system partitions the image into overlapping multi-scale windows and aggregates spatial probabilities.</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Upload & Slicing Control */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative min-h-[220px]">
                    <input 
                      type="file" 
                      accept="image/*"
                      ref={singleInputRef}
                      onChange={handleGridUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    {gridPreview ? (
                      <div className="w-full flex flex-col items-center justify-center relative">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setGridImage(null);
                            setGridPreview(null);
                            setGridResult(null);
                          }}
                          className="absolute -top-3 -right-3 p-1.5 bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 rounded-full shadow-md z-30 transition-all"
                          title="Remove Image"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                        
                        <div className="relative inline-block border border-slate-200 rounded-xl overflow-hidden shadow-md max-w-[500px]">
                          <img 
                            src={gridPreview} 
                            alt="Grid Input" 
                            className="max-h-[360px] object-contain"
                          />
                          
                          {/* Grid Overlay */}
                          {gridResult && gridResult.grid_size && (
                            <div 
                              className="absolute inset-0 grid overflow-hidden"
                              style={{
                                gridTemplateRows: `repeat(${gridResult.grid_size.rows}, 1fr)`,
                                gridTemplateColumns: `repeat(${gridResult.grid_size.cols}, 1fr)`
                              }}
                            >
                              {gridResult.predictions.map((cell, idx) => (
                                <div
                                  key={idx}
                                  onMouseEnter={() => setHoveredGridCell(cell)}
                                  onMouseLeave={() => setHoveredGridCell(null)}
                                  className={`border transition-all duration-100 cursor-crosshair ${
                                    cell.is_low_confidence ? 'border-amber-400 border-dashed' : 'border-white/10'
                                  }`}
                                  style={{
                                    backgroundColor: hoveredGridCell === cell 
                                      ? `${CLASS_COLORS[cell.prediction]}60` 
                                      : `${CLASS_COLORS[cell.prediction]}30`
                                  }}
                                ></div>
                              ))}
                            </div>
                          )}
                        </div>
                        <p className="text-xs text-slate-500 mt-2 font-mono">{gridImage?.name}</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <Upload className="w-8 h-8 text-blue-500 mx-auto" />
                        <div>
                          <p className="text-sm font-bold text-slate-700">Upload Regional Satellite Image</p>
                          <p className="text-xs text-slate-400">Supports large composite satellite imagery</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {gridImage && (
                    <button
                      onClick={triggerGridPredict}
                      disabled={analyzingGrid}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2 text-xs disabled:opacity-50"
                    >
                      {analyzingGrid ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Slicing & Aggregating Overlapping Spatial Probabilities...
                        </>
                      ) : (
                        <>
                          <Layers className="w-4 h-4" />
                          Run Overlapping Grid Classification
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Grid Cell Inspector Side Panel */}
                <div className="border border-slate-200 rounded-2xl bg-slate-50/50 p-6 flex flex-col justify-between">
                  {hoveredGridCell ? (
                    <div className="space-y-4">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Cell Inspector</h4>
                      <div className="space-y-1">
                        <h5 className="text-2xl font-extrabold text-slate-900 font-display flex items-center gap-2">
                          {hoveredGridCell.prediction}
                          <span 
                            className="w-3.5 h-3.5 rounded-full"
                            style={{ backgroundColor: CLASS_COLORS[hoveredGridCell.prediction] }}
                          ></span>
                        </h5>
                        <p className="text-xs text-blue-600 font-mono font-bold">
                          Confidence: {hoveredGridCell.confidence.toFixed(1)}%
                        </p>
                        {hoveredGridCell.is_low_confidence && (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 bg-amber-100 text-amber-800 rounded border border-amber-200">
                            <AlertTriangle className="w-3 h-3 text-amber-600" /> Uncertain Prediction
                          </span>
                        )}
                      </div>
                      
                      {hoveredGridCell.top_predictions && (
                        <div className="space-y-2 pt-2 border-t border-slate-200">
                          <p className="text-[10px] font-bold text-slate-400 uppercase">Cell Candidates</p>
                          {hoveredGridCell.top_predictions.map((cp, idx) => (
                            <div key={idx} className="flex justify-between text-xs font-semibold">
                              <span>{cp.class}</span>
                              <span className="font-mono text-slate-500">{cp.confidence.toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : gridResult ? (
                    <div className="space-y-4">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Grid Mapping Summary</h4>
                      <div className="space-y-2">
                        <p className="text-xs text-slate-600 font-medium">
                          Dimensions: {gridResult.image_size?.width} &times; {gridResult.image_size?.height} px
                        </p>
                        <p className="text-xs text-slate-600 font-medium">
                          Total Cells: {gridResult.predictions.length} patches ({gridResult.grid_size?.rows} rows &times; {gridResult.grid_size?.cols} cols)
                        </p>
                        <p className="text-xs text-slate-500 pt-2 border-t border-slate-200">
                          Hover over any cell on the image grid to inspect exact class predictions and confidence values.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-6 space-y-3">
                      <HelpCircle className="w-8 h-8 text-slate-400 mx-auto" />
                      <h4 className="text-xs font-bold text-slate-700">Grid Classifier Ready</h4>
                      <p className="text-[11px] text-slate-400">Upload an image and run grid classification to map regional land cover.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: CHANGE DETECTION */}
          {activeTab === 'change' && (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-8 animate-fade-in">
              <div className="border-b border-slate-100 pb-4">
                <h3 className="text-lg font-bold text-slate-800 font-display">Historical GIS Change Detection</h3>
                <p className="text-xs text-slate-500">Upload two satellite images of the same location from different time periods</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                
                {/* Previous Image */}
                <div className="space-y-3">
                  <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                    Historical (Previous) Image
                  </h4>
                  <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative min-h-[220px]">
                    <input 
                      type="file" 
                      accept="image/*"
                      ref={oldInputRef}
                      onChange={(e) => {
                        const f = e.target.files[0];
                        if (f) {
                          setOldImage(f);
                          setOldPreview(URL.createObjectURL(f));
                          setChangeResult(null);
                        }
                      }}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    {oldPreview ? (
                      <div className="relative">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setOldImage(null);
                            setOldPreview(null);
                            setChangeResult(null);
                            if (oldInputRef.current) oldInputRef.current.value = "";
                          }}
                          className="absolute -top-3 -right-3 p-1 bg-red-100 hover:bg-red-200 text-red-600 rounded-full shadow z-20"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                        <img src={oldPreview} alt="Old Preview" className="max-h-[160px] object-contain rounded-lg border shadow-sm" />
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Upload className="w-6 h-6 text-slate-400 mx-auto" />
                        <p className="text-xs font-bold text-slate-500">Upload Historical Image</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Current Image */}
                <div className="space-y-3">
                  <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    Current (Recent) Image
                  </h4>
                  <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative min-h-[220px]">
                    <input 
                      type="file" 
                      accept="image/*"
                      ref={newInputRef}
                      onChange={(e) => {
                        const f = e.target.files[0];
                        if (f) {
                          setNewImage(f);
                          setNewPreview(URL.createObjectURL(f));
                          setChangeResult(null);
                        }
                      }}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    {newPreview ? (
                      <div className="relative">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setNewImage(null);
                            setNewPreview(null);
                            setChangeResult(null);
                            if (newInputRef.current) newInputRef.current.value = "";
                          }}
                          className="absolute -top-3 -right-3 p-1 bg-red-100 hover:bg-red-200 text-red-600 rounded-full shadow z-20"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                        <img src={newPreview} alt="New Preview" className="max-h-[160px] object-contain rounded-lg border shadow-sm" />
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Upload className="w-6 h-6 text-slate-400 mx-auto" />
                        <p className="text-xs font-bold text-slate-500">Upload Current Image</p>
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {oldImage && newImage && (
                <button
                  onClick={triggerChangeDetection}
                  disabled={detectingChange}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 text-xs"
                >
                  {detectingChange ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Evaluating Land Cover Transitions...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      Run Change Detection Algorithm
                    </>
                  )}
                </button>
              )}

              {/* Analysis Result */}
              {changeResult && (
                <div className="border border-slate-200 rounded-2xl bg-slate-50/50 p-6 space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                    <div>
                      <h4 className="text-base font-bold text-slate-800">Environmental Impact Analysis</h4>
                      <p className="text-xs text-slate-500">Transition evaluation based on EuroSAT taxonomy rules</p>
                    </div>
                    
                    <span className={`px-4 py-1.5 rounded-full text-xs font-bold border ${
                      changeResult.change_detected 
                        ? 'bg-rose-50 text-rose-800 border-rose-200' 
                        : 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    }`}>
                      {changeResult.change_detected ? "CHANGE DETECTED" : "STABLE LAND USE"}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    
                    <div className="grid grid-cols-2 gap-4 bg-white p-4 rounded-xl border border-slate-200">
                      <div className="space-y-1">
                        <div className="aspect-square relative rounded-lg border overflow-hidden">
                          <img src={`${API_BASE}${changeResult.old_image_url}`} alt="Old" className="w-full h-full object-cover" />
                        </div>
                        <div className="text-center">
                          <p className="text-[10px] text-slate-400 font-bold uppercase">Historical</p>
                          <p className="text-xs font-bold text-slate-700">{changeResult.old_class}</p>
                          <p className="text-[10px] font-mono text-slate-500">({changeResult.old_confidence.toFixed(1)}%)</p>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <div className="aspect-square relative rounded-lg border overflow-hidden">
                          <img src={`${API_BASE}${changeResult.new_image_url}`} alt="New" className="w-full h-full object-cover" />
                        </div>
                        <div className="text-center">
                          <p className="text-[10px] text-slate-400 font-bold uppercase">Recent</p>
                          <p className="text-xs font-bold text-slate-700">{changeResult.new_class}</p>
                          <p className="text-[10px] font-mono text-slate-500">({changeResult.new_confidence.toFixed(1)}%)</p>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-2">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Identified Impact</span>
                        <h4 className="text-2xl font-black text-slate-900 tracking-tight">{changeResult.impact}</h4>
                        <p className="text-xs text-slate-600 leading-relaxed pt-1">
                          {changeResult.change_detected ? (
                            `Class transition identified from "${changeResult.old_class}" to "${changeResult.new_class}".`
                          ) : (
                            "No significant class shifts identified. The terrain indicates stable cover characteristics."
                          )}
                        </p>
                      </div>
                    </div>

                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 5: HISTORY LOGS */}
          {activeTab === 'history' && (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6 animate-fade-in">
              <div className="flex flex-col md:flex-row gap-4 items-center justify-between border-b border-slate-100 pb-4">
                <div className="relative w-full md:w-80">
                  <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search filename or class..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="flex gap-3 w-full md:w-auto">
                  <div className="relative w-full md:w-48">
                    <Filter className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                    <select
                      value={classFilter}
                      onChange={(e) => setClassFilter(e.target.value)}
                      className="w-full pl-9 pr-4 py-2.5 border border-slate-200 rounded-lg text-xs appearance-none focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white"
                    >
                      <option value="">All Categories</option>
                      {CLASSES.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl shadow-sm">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-400 font-semibold bg-slate-50">
                      <th className="py-3 px-4">Image</th>
                      <th className="py-3 px-4">Grad-CAM Saliency</th>
                      <th className="py-3 px-4">Filename</th>
                      <th className="py-3 px-4">Category</th>
                      <th className="py-3 px-4">Confidence</th>
                      <th className="py-3 px-4">Timestamp</th>
                      <th className="py-3 px-4 text-center">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredHistory.map((row) => (
                      <tr key={row.id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="py-3 px-4">
                          <img 
                            src={`${API_BASE}${row.image_url}`} 
                            alt="Original" 
                            className="w-12 h-12 object-cover rounded-md border border-slate-200"
                            onError={(e) => { e.target.src = 'https://placehold.co/100x100?text=Satellite'; }}
                          />
                        </td>
                        <td className="py-3 px-4">
                          {row.heatmap_url ? (
                            <img 
                              src={`${API_BASE}${row.heatmap_url}`} 
                              alt="Heatmap" 
                              className="w-12 h-12 object-cover rounded-md border border-slate-200"
                              onError={(e) => { e.target.src = 'https://placehold.co/100x100?text=Grad-CAM'; }}
                            />
                          ) : (
                            <span className="text-[10px] text-slate-400 font-mono">N/A</span>
                          )}
                        </td>
                        <td className="py-3 px-4 font-semibold text-slate-700">{row.image_name}</td>
                        <td className="py-3 px-4">
                          <span className="px-2.5 py-1 rounded-full text-[10px] font-bold" style={{ backgroundColor: `${CLASS_COLORS[row.prediction]}20`, color: CLASS_COLORS[row.prediction] }}>
                            {row.prediction}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono font-semibold text-blue-600">{row.confidence.toFixed(1)}%</td>
                        <td className="py-3 px-4 text-slate-500">{new Date(row.timestamp).toLocaleString()}</td>
                        <td className="py-3 px-4 text-center">
                          <div className="flex items-center justify-center gap-3">
                            <a
                              href={`${API_BASE}/download_report/${row.id}`}
                              className="text-slate-500 hover:text-blue-600 inline-flex items-center gap-1 font-bold"
                              title="Download PDF Report"
                            >
                              <FileText className="w-4 h-4" />
                            </a>
                            <button
                              onClick={() => deleteHistoryRecord(row.id)}
                              className="text-slate-400 hover:text-red-600 transition-colors"
                              title="Delete Record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                    {filteredHistory.length === 0 && (
                      <tr>
                        <td colSpan="7" className="py-8 text-center text-slate-400 font-semibold">
                          No matching prediction records found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 6: ANALYTICS & VALIDATION PANEL */}
          {activeTab === 'analytics' && (
            <div className="space-y-8 animate-fade-in">
              
              {/* External Validation Controller Banner */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-slate-800 font-display">Out-of-Distribution External Dataset Evaluator</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Evaluates the model separately on images stored in <code className="bg-slate-100 px-1 rounded">external_validation/</code> class folders (outside EuroSAT).
                  </p>
                </div>
                <button
                  onClick={triggerEvalExternal}
                  disabled={evaluatingExt}
                  className="bg-purple-700 hover:bg-purple-800 text-white font-bold px-5 py-2.5 rounded-xl text-xs transition-colors flex items-center gap-2 disabled:opacity-50 whitespace-nowrap"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${evaluatingExt ? 'animate-spin' : ''}`} />
                  Run External Dataset Evaluation
                </button>
              </div>

              {/* Distribution Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Class Distribution Pie Chart */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                  <h3 className="text-base font-bold text-slate-800 font-display">Logged Class Coverage Distribution</h3>
                  <div className="aspect-square max-h-[340px] mx-auto flex items-center justify-center">
                    {analytics && Object.keys(analytics.class_distribution || {}).length > 0 ? (
                      <Pie 
                        data={getPieChartData()} 
                        options={{
                          plugins: {
                            legend: {
                              position: 'bottom',
                              labels: { boxWidth: 12, font: { size: 10 } }
                            }
                          }
                        }}
                      />
                    ) : (
                      <p className="text-xs text-slate-400 font-semibold">No distribution data. Classify some images to render chart!</p>
                    )}
                  </div>
                </div>

                {/* Activity Trends */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                  <h3 className="text-base font-bold text-slate-800 font-display">Classification Activity Trends</h3>
                  <div className="aspect-video flex items-center justify-center">
                    {analytics && Object.keys(analytics.trends || {}).length > 0 ? (
                      <Line 
                        data={getLineChartData()} 
                        options={{
                          responsive: true,
                          plugins: { legend: { display: false } },
                          scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
                        }}
                      />
                    ) : (
                      <p className="text-xs text-slate-400 font-semibold">No classification logs. Perform classifications to plot trend!</p>
                    )}
                  </div>
                </div>

              </div>

              {/* Epoch History Bar Chart */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-base font-bold text-slate-800 font-display">Production Training Accuracy Curve</h3>
                <div className="aspect-[3/1] max-h-[280px] flex items-center justify-center">
                  {analytics && analytics.model_metrics?.history ? (
                    <Bar 
                      data={getBarChartData()} 
                      options={{
                        responsive: true,
                        scales: { y: { min: 0, max: 1 } }
                      }}
                    />
                  ) : (
                    <p className="text-xs text-slate-400 font-semibold">No model history metrics. Run production training to render history.</p>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
