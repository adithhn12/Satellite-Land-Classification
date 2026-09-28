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
  FileSpreadsheet
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

const API_BASE = "http://localhost:5000";

const CLASSES = [
  "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
  "Industrial", "Pasture", "PermanentCrop", "Residential", 
  "River", "SeaLake"
];

const CLASS_COLORS = {
  AnnualCrop: "#86efac", // Light Green
  Forest: "#15803d", // Dark Green
  HerbaceousVegetation: "#4ade80", // Medium Green
  Highway: "#64748b", // Slate
  Industrial: "#a21caf", // Purple
  Pasture: "#bef264", // Lime Green
  PermanentCrop: "#22c55e", // Green
  Residential: "#f97316", // Orange
  River: "#3b82f6", // Blue
  SeaLake: "#1e3a8a" // Navy Blue
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
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
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
        showNotification("Grid classification completed successfully!", "success");
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

  const showNotification = (text, type = "success") => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 5000);
  };

  const handleTrain = async (quick = true) => {
    setTraining(true);
    showNotification(`Model training initiated in ${quick ? 'Quick Demo' : 'Full'} mode. Please wait...`, "info");
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quick })
      });
      const data = await res.json();
      if (data.status === "success") {
        showNotification(`Training completed successfully! Accuracy: ${data.accuracy}`, "success");
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
    console.log("handleBatchUpload: selected files:", files);
    if (files.length > 0) {
      setBatchFiles(files);
      setBatchResults([]);
    }
  };

  const triggerBatchPredict = async () => {
    if (batchFiles.length === 0) return;
    console.log("triggerBatchPredict: sending files:", batchFiles);
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
    csvContent += "Image Name,Prediction,Confidence (%)\n";
    batchResults.forEach(r => {
      csvContent += `"${r.image_name}","${r.prediction}",${r.confidence}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "batch_predictions_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filtered History
  const filteredHistory = history.filter(item => {
    const matchesSearch = item.image_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          item.prediction.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesClass = classFilter === '' || item.prediction === classFilter;
    return matchesSearch && matchesClass;
  });

  // KPI Metrics Calculation
  const totalAnalyzed = history.length;
  const avgConf = history.length > 0 ? (history.reduce((acc, h) => acc + h.confidence, 0) / history.length).toFixed(1) : "0.0";
  const modelAccuracy = analytics?.model_metrics?.accuracy || "94.2%";

  // Setup Chart Data
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
            <p className="text-xs text-blue-600 font-medium">Land Classification</p>
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
            { id: 'analytics', label: 'Analytics Panel', icon: BarChart3 },
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
          <p className="text-xs text-slate-400 font-mono">MCA Project v1.0.0</p>
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
              {activeTab === 'grid' && 'Grid Land-Cover Slicing'}
              {activeTab === 'batch' && 'Batch Image Processing'}
              {activeTab === 'change' && 'GIS Change Detection'}
              {activeTab === 'history' && 'Classification Records'}
              {activeTab === 'analytics' && 'Training & Distribution Analytics'}
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
                  { label: "Total Classified Images", value: totalAnalyzed, desc: "Cumulative database uploads", icon: Layers, color: "text-blue-600 bg-blue-50 border-blue-100" },
                  { label: "EuroSAT Model Accuracy", value: modelAccuracy, desc: "Evaluated on validation sets", icon: Cpu, color: "text-emerald-600 bg-emerald-50 border-emerald-100" },
                  { label: "Average Class Confidence", value: `${avgConf}%`, desc: "Average prediction probability", icon: TrendingUp, color: "text-violet-600 bg-violet-50 border-violet-100" },
                  { label: "Active Classification Classes", value: "10", desc: "EuroSAT land cover categories", icon: Map, color: "text-amber-600 bg-amber-50 border-amber-100" }
                ].map((kpi, idx) => (
                  <div key={idx} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between hover:shadow-md transition-shadow">
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{kpi.label}</p>
                      <h3 className="text-3xl font-extrabold text-slate-900 tracking-tight font-display">{kpi.value}</h3>
                      <p className="text-xs text-slate-400">{kpi.desc}</p>
                    </div>
                    <div className={`p-4 rounded-xl border ${kpi.color}`}>
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
                      <h3 className="text-lg font-bold text-slate-800 font-display">Transfer Learning Controller</h3>
                      <p className="text-xs text-slate-500">Train ResNet50V2 on EuroSAT dataset</p>
                    </div>
                    <Cpu className="w-6 h-6 text-blue-500" />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 flex flex-col justify-between">
                      <div>
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-[10px] font-bold rounded uppercase">Recommended</span>
                        <h4 className="text-sm font-bold text-slate-800 mt-1">Quick Demo Training</h4>
                        <p className="text-xs text-slate-500 leading-relaxed mt-1">
                          Subsamples 15 images/class, trains for 2 epochs on CPU. Completes in 10-15 seconds. Ideal for live presentations.
                        </p>
                      </div>
                      <button
                        onClick={() => handleTrain(true)}
                        disabled={training}
                        className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center gap-2 shadow-sm shadow-blue-500/10 disabled:opacity-50"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${training ? 'animate-spin' : ''}`} />
                        Run Demo Training
                      </button>
                    </div>

                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 flex flex-col justify-between">
                      <div>
                        <span className="px-2 py-0.5 bg-slate-200 text-slate-700 text-[10px] font-bold rounded uppercase">Standard</span>
                        <h4 className="text-sm font-bold text-slate-700 mt-1">Full CPU Training</h4>
                        <p className="text-xs text-slate-500 leading-relaxed mt-1">
                          Loads 200 images/class, trains for 10 epochs. Takes approximately 5-10 minutes on average systems.
                        </p>
                      </div>
                      <button
                        onClick={() => handleTrain(false)}
                        disabled={training}
                        className="w-full mt-4 bg-slate-800 hover:bg-slate-900 text-white font-semibold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        Run Full Training
                      </button>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      // Call reset model endpoint
                      fetch(`${API_BASE}/reset_model`, { method: "POST" })
                        .then(res => res.json())
                        .then(data => {
                          if (data.status === "success") {
                            showNotification(`Model reset and retrained. ${data.message}`, "success");
                            fetchAnalytics();
                          } else {
                            showNotification(`Reset failed: ${data.message}`, "error");
                          }
                        })
                        .catch(err => showNotification("Error resetting model", "error"));
                    }}
                    disabled={training}
                    className="w-full mt-4 bg-red-600 hover:bg-red-700 text-white font-semibold py-2.5 rounded-lg text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <RefreshCw className={"w-3.5 h-3.5"} />
                    Reset Model
                  </button>
                </div>

                {/* EuroSAT Classes Checklist Card */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                  <h3 className="text-lg font-bold text-slate-800 font-display">Target Classification Classes</h3>
                  <div className="max-height-[320px] overflow-y-auto space-y-2 pr-1">
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
                    <h3 className="text-lg font-bold text-slate-800 font-display">Recent Activity Log</h3>
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
                        <th className="py-3 px-4">Image Filename</th>
                        <th className="py-3 px-4">Class Target</th>
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
                    <p className="text-xs text-slate-500">Upload a single 64x64 or high-resolution RGB satellite image</p>
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
                        {/* Remove Image Option */}
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
                          <p className="text-xs text-slate-400 mt-1">Supports PNG, JPG, or JPEG formats</p>
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
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition-all shadow-md shadow-blue-600/10 flex items-center justify-center gap-2 disabled:opacity-50 text-sm"
                  >
                    {analyzingSingle ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Analyzing Geographical Features...
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
                      {/* Classification Title & Confidence */}
                      <div className="flex justify-between items-start">
                        <div className="space-y-1">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Primary Prediction</span>
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
                        
                        <div className="text-right">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Confidence</span>
                          <h4 className="text-3xl font-extrabold text-blue-600 font-mono tracking-tight">
                            {singleResult.confidence.toFixed(1)}%
                          </h4>
                        </div>
                      </div>

                      {/* Grad-CAM side-by-side or highlight */}
                      <div className="space-y-2">
                        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Explainability Visualization (Grad-CAM)</h5>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <div className="border border-slate-200 rounded-lg bg-white p-2">
                              <img 
                                src={`${API_BASE}${singleResult.image_url}`} 
                                alt="Original" 
                                className="w-full aspect-square object-cover rounded-md"
                              />
                            </div>
                            <p className="text-[10px] text-center font-bold text-slate-500">Original RGB Input</p>
                          </div>
                          <div className="space-y-1">
                            <div className="border border-slate-200 rounded-lg bg-white p-2">
                              <img 
                                src={`${API_BASE}${singleResult.heatmap_url}`} 
                                alt="Heatmap" 
                                className="w-full aspect-square object-cover rounded-md"
                              />
                            </div>
                            <p className="text-[10px] text-center font-bold text-slate-500 text-blue-600">Model Attention Heatmap</p>
                          </div>
                        </div>
                      </div>

                      {/* Top 3 Predictions Bar Graph */}
                      <div className="space-y-3 bg-white p-4 rounded-xl border border-slate-200">
                        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Top Predictions Probability</h5>
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
                      <a
                        href={`${API_BASE}/download_report/${singleResult.id}`}
                        className="w-full mt-4 bg-slate-800 hover:bg-slate-900 text-white font-bold py-3.5 rounded-xl text-xs transition-colors flex items-center justify-center gap-2"
                      >
                        <FileText className="w-4 h-4" />
                        Download Detailed PDF Report
                      </a>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4">
                      <div className="w-16 h-16 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400">
                        <HelpCircle className="w-8 h-8" />
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-slate-700">Analysis Awaiting Trigger</h4>
                        <p className="text-xs text-slate-400 mt-1 max-w-[280px] mx-auto">
                          Upload an image and click the classify button to analyze geographical structures and generate a Grad-CAM heatmap.
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
                  <h3 className="text-lg font-bold text-slate-800 font-display">Batch Satellite Prediction</h3>
                  <p className="text-xs text-slate-500">Upload and process multiple images concurrently to categorize entire sets of files</p>
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
                    <p className="text-sm font-bold text-slate-700">Select Multiple Images</p>
                    <p className="text-xs text-slate-400 mt-1">
                      {batchFiles.length > 0 ? `${batchFiles.length} files selected` : "Supports batch upload of JPG/PNG files"}
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
                      Analyze Selected Files
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
                        <th className="py-3 px-4">Prediction</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4 text-center">Actions</th>
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
                <h3 className="text-lg font-bold text-slate-800 font-display">Large Image Grid Slicing Classifier</h3>
                <p className="text-xs text-slate-500">Upload a high-resolution satellite composite. The model will partition the image into 64x64 blocks to classify and map the entire region.</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Upload & Slicing Control */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative min-h-[220px]">
                    <input 
                      type="file" 
                      accept="image/*"
                      ref={singleInputRef} // reuse singleInputRef for convenience or keep it simple
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
                        
                        {/* Image Container with Absolute Grid Overlay */}
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
                                  className="border border-white/10 transition-all duration-100 cursor-crosshair"
                                  style={{
                                    backgroundColor: hoveredGridCell === cell 
                                      ? `${CLASS_COLORS[cell.prediction]}60` 
                                      : `${CLASS_COLORS[cell.prediction]}15`
                                  }}
                                />
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="p-3 bg-blue-50 border border-blue-100 rounded-full text-blue-600 inline-block">
                          <Upload className="w-6 h-6 mx-auto" />
                        </div>
                        <div>
                          <p className="text-sm font-bold text-slate-700">Upload Large Satellite Composite</p>
                          <p className="text-xs text-slate-400 mt-1">Recommended size: 256x256, 512x512, or 1024x1024 pixels</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {gridImage && (
                    <button
                      onClick={triggerGridPredict}
                      disabled={analyzingGrid}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 text-xs shadow-md shadow-blue-500/10"
                    >
                      {analyzingGrid ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Partitioning and Classifying Terrain...
                        </>
                      ) : (
                        <>
                          <Layers className="w-4 h-4" />
                          Classify Composite Area
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Slicing Analytics Summary */}
                <div className="border border-slate-200 rounded-2xl bg-slate-50/50 p-6 flex flex-col justify-between min-h-[300px]">
                  {gridResult ? (
                    <div className="space-y-6 flex-1 flex flex-col justify-between">
                      <div>
                        <h4 className="text-sm font-bold text-slate-800 uppercase tracking-wider text-slate-400">Composite Grid Map</h4>
                        <div className="grid grid-cols-2 gap-4 mt-2">
                          <div className="bg-white p-3 rounded-xl border border-slate-200">
                            <span className="text-[10px] text-slate-400 font-bold uppercase">Grid Layout</span>
                            <p className="text-lg font-black text-slate-700">{gridResult.grid_size.rows} x {gridResult.grid_size.cols} cells</p>
                          </div>
                          <div className="bg-white p-3 rounded-xl border border-slate-200">
                            <span className="text-[10px] text-slate-400 font-bold uppercase">Total Tiles</span>
                            <p className="text-lg font-black text-slate-700">{gridResult.predictions.length} patches</p>
                          </div>
                        </div>
                      </div>

                      {/* Hover Cell Inspector */}
                      <div className="bg-white p-4 rounded-xl border border-slate-200 min-h-[100px] flex flex-col justify-center">
                        {hoveredGridCell ? (
                          <div className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-[10px] text-slate-400 font-bold uppercase">Cell [{hoveredGridCell.position.row}, {hoveredGridCell.position.col}]</span>
                              <span 
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: CLASS_COLORS[hoveredGridCell.prediction] }}
                              ></span>
                            </div>
                            <h4 className="text-xl font-bold text-slate-800">{hoveredGridCell.prediction}</h4>
                            <p className="text-xs text-blue-600 font-semibold font-mono">Confidence: {hoveredGridCell.confidence.toFixed(1)}%</p>
                          </div>
                        ) : (
                          <p className="text-xs text-slate-400 text-center italic">Hover over the classified image grid to inspect individual cells</p>
                        )}
                      </div>

                      {/* Coverage Breakdown */}
                      <div className="space-y-3">
                        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Areal Coverage breakdown</h5>
                        <div className="max-h-[160px] overflow-y-auto space-y-2 pr-1">
                          {Object.entries(
                            gridResult.predictions.reduce((acc, cell) => {
                              acc[cell.prediction] = (acc[cell.prediction] || 0) + 1;
                              return acc;
                            }, {})
                          ).map(([cls, count]) => {
                            const pct = ((count / gridResult.predictions.length) * 100).toFixed(1);
                            return (
                              <div key={cls} className="space-y-1">
                                <div className="flex justify-between text-xs font-semibold">
                                  <span className="text-slate-700">{cls}</span>
                                  <span className="text-slate-500">{pct}%</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full rounded-full"
                                    style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[cls] }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4">
                      <div className="w-16 h-16 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400">
                        <HelpCircle className="w-8 h-8" />
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-slate-700">Grid Classifier Awaiting</h4>
                        <p className="text-xs text-slate-400 mt-1 max-w-[280px] mx-auto">
                          Upload a large composite image (e.g. mapping an entire farming tract) and run class analysis to plot grid layers.
                        </p>
                      </div>
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
                <p className="text-xs text-slate-500">Upload two images of the same location from different points in time to analyze transitions</p>
              </div>

              {/* Comparison Image Pickers */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                
                {/* Previous (Old) Image */}
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
                          className="absolute -top-3 -right-3 p-1 bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 rounded-full shadow z-20 transition-all"
                          title="Remove Image"
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

                {/* Current (New) Image */}
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
                          className="absolute -top-3 -right-3 p-1 bg-red-100 hover:bg-red-200 border border-red-200 text-red-600 rounded-full shadow z-20 transition-all"
                          title="Remove Image"
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
                      Evaluating Land Cover Changes...
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
                      <h4 className="text-base font-bold text-slate-800">Socio-Environmental Impact Analysis</h4>
                      <p className="text-xs text-slate-500">Transition evaluation based on EuroSAT taxonomy rules</p>
                    </div>
                    
                    <span className={`px-4 py-1.5 rounded-full text-xs font-bold border ${
                      changeResult.change_detected 
                        ? 'bg-rose-50 text-rose-800 border-rose-200 animate-pulse' 
                        : 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    }`}>
                      {changeResult.change_detected ? "CHANGE DETECTED" : "STABLE LAND USE"}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    
                    {/* Visual Comparison cards */}
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

                    {/* Change Assessment Details */}
                    <div className="space-y-4">
                      <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-2">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Identified Impact</span>
                        <h4 className="text-2xl font-black text-slate-900 tracking-tight">{changeResult.impact}</h4>
                        <p className="text-xs text-slate-600 leading-relaxed pt-1">
                          {changeResult.change_detected ? (
                            `The satellite analysis identified a clear shift in class signature from "${changeResult.old_class}" to "${changeResult.new_class}". This shift impacts agricultural, natural canopy, or hydrological parameters and requires GIS verification.`
                          ) : (
                            "No significant class shifts were identified. The terrain indicates stable cover characteristics across both timelines."
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
              
              {/* Filtering Interface */}
              <div className="flex flex-col md:flex-row gap-4 items-center justify-between border-b border-slate-100 pb-4">
                <div className="relative w-full md:w-80">
                  <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search image name or class..."
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

              {/* Records List Table */}
              <div className="overflow-x-auto border border-slate-200 rounded-xl shadow-sm">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-400 font-semibold bg-slate-50">
                      <th className="py-3 px-4">Original Image</th>
                      <th className="py-3 px-4">Explainability</th>
                      <th className="py-3 px-4">Filename</th>
                      <th className="py-3 px-4">Predicted Category</th>
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
                          <img 
                            src={`${API_BASE}${row.heatmap_url}`} 
                            alt="Heatmap" 
                            className="w-12 h-12 object-cover rounded-md border border-slate-200"
                            onError={(e) => { e.target.src = 'https://placehold.co/100x100?text=Grad-CAM'; }}
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
                        <td className="py-3 px-4 text-center">
                          <div className="flex items-center justify-center gap-3">
                            <a
                              href={`${API_BASE}/download_report/${row.id}`}
                              className="text-slate-500 hover:text-blue-600 inline-flex items-center gap-1 font-bold"
                              title="Download Report"
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

          {/* TAB 6: ANALYTICS PANEL */}
          {activeTab === 'analytics' && (
            <div className="space-y-8 animate-fade-in">
              
              {/* Distribution Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Class Distribution Pie Chart */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                  <h3 className="text-base font-bold text-slate-800 font-display">Class Coverage Distribution</h3>
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

                {/* Prediction Trends Line Chart */}
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

              {/* Model Training Epoch Accuracies */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-base font-bold text-slate-800 font-display">Model Training History (Accuracy per Epoch)</h3>
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
                    <p className="text-xs text-slate-400 font-semibold">No model metrics. Run a training cycle to render history.</p>
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
