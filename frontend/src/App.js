import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import animation from './animation.gif';
import icon from './delete.png';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import { GoogleLogin } from '@react-oauth/google';
import { GoogleOAuthProvider } from '@react-oauth/google';
import projectNameGenerator from 'project-name-generator';

const availableEffects = ['Reverb', 'Pitch Shifter'];
const responseMessage = "Login successful!";
const errorMessage = "Login failed!";


const App = () => {
  const [instrumentStems, setInstrumentStems] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [waveformImages, setWaveformImages] = useState([]);
  const [audioIndexes, setAudioIndexes] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [originalStems, setOriginalStems] = useState([]);
  const [exportPopupOpen, setExportPopupOpen] = useState(false); // State to control export popup
  const [selectedTracks, setSelectedTracks] = useState([]); // State to store selected tracks for export
  const [trackPans, setTrackPans] = useState([]); // State to store track pans
  const audioRefs = useRef([]);
  const [instrumentNames, setInstrumentNames] = useState(['Voice', 'Drums', 'Bass', 'Other']);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [token, setToken] = useState(null);
  const [resetTrigger, setResetTrigger] = useState(false);
  const fileInputRef = useRef(null);
  const [selectedEffectsList, setSelectedEffectsList] = useState([]); //new proj
  const [projectName, setProjectName] = useState(''); //new proj
  const [projects, setProjects] = useState([]);
  const [instrumentVolumes, setInstrumentVolumes] = useState([]);
  const [instrumentPans, setInstrumentPans] = useState([]);

  
const fetchProjectsByCredential = async () => {
  if (token && token.credential) {
    try {
      const credential = token.credential.slice(0, 255);
      const response = await fetch(`http://127.0.0.1:8000/api/get_projects_by_credential/?credential=${credential}`, {
        method: 'GET'
      });
      if (response.ok) {
        const data = await response.json();
        setProjects(data);
        console.log("Project names fetched!");
      } else {
        console.error('Error fetching projects:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }
};

// Fetch projects when the component mounts or when the token changes
useEffect(() => {
  fetchProjectsByCredential();
}, [token]);


  const updateSelectedEffects = (index, effects) => {
    const newEffectsList = [...selectedEffectsList];
    newEffectsList[index] = effects;
    setSelectedEffectsList(newEffectsList);
  };
  
  const generateRandomDashedProjectName = () => {
    return projectNameGenerator().dashed;
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const uploadFile = async () => {
    setLoading(true); // Set loading state to true when uploading starts
    const formData = new FormData();
    formData.append('audio_file', selectedFile);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/separate_audio/', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Separated files:', data.separated_files);
        console.log('Waveform images:', data.waveform_images);
        

        const newProjectName = generateRandomDashedProjectName();
        setProjectName(newProjectName);
        console.log('Generated Project Name:', newProjectName);
        
        setInstrumentStems(data.separated_files);
        setOriginalStems([...data.separated_files]);
        setWaveformImages(data.waveform_images);
        setAudioIndexes(Array.from({ length: data.separated_files.length }, (_, i) => i));
        setTrackPans(Array.from({ length: data.separated_files.length }, () => 0)); // Initialize track pans
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false); // Set loading state to false when upload is completed
    }
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
    if (isPlaying) {
      audioRefs.current.forEach((audio, index) => {
        if (!audioRefs.current[index].muted) {
          audio.pause();
        }
      });
    } else {
      audioRefs.current.forEach((audio) => audio.play());
    }
  };

  const handleTimeUpdate = () => {
    const currentTime = audioRefs.current.reduce((minTime, audio) => Math.min(minTime, audio.currentTime), Infinity);
    setCurrentTime(currentTime);
  };

  const handleProgressClick = (event) => {
    const clickedTime = (event.nativeEvent.offsetX / event.target.offsetWidth) * audioRefs.current[0].duration;
    audioRefs.current.forEach((audio) => audio.currentTime = clickedTime);
    setCurrentTime(clickedTime);
  };

  const handleMuteToggle = (index) => {
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.muted = !audioRef.muted;
    }
  };

  const handleSoloToggle = (index) => {
    audioRefs.current.forEach((audio, i) => {
      if (i !== index) {
        audio.muted = !audio.muted;
      }
    });
  };

  const handleEffectAdd = async (index, selectedEffect) => {
    const audioBlob = await fetch(`data:audio/wav;base64,${instrumentStems[index]}`).then(response => response.blob());
    const formData = new FormData();
    formData.append('audio_file', audioBlob);
    formData.append('effects', selectedEffect);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/process_audio_with_effects/', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Processed audio:', data.processed_audio);
        const newInstrumentStems = [...instrumentStems];
        newInstrumentStems[index] = data.processed_audio;
        setInstrumentStems(newInstrumentStems);
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
    
    audioRefs.current.forEach((audio) => audio.currentTime = 0);
    setCurrentTime(0);

  };

  const handleEffectAdd2 = async (index, selectedEffects) => {
    const effectsString = selectedEffects.join(','); // Convert selected effects to a comma-separated string
    console.log(effectsString);
    const audioBlob = await fetch(`data:audio/wav;base64,${originalStems[index]}`).then(response => response.blob());
    const formData = new FormData();
    formData.append('audio_file', audioBlob);
    formData.append('effects', effectsString); // Send the comma-separated list of effects
    try {
      const response = await fetch('http://127.0.0.1:8000/api/process_audio_with_effects/', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Processed audio:', data.processed_audio);
        const newInstrumentStems = [...instrumentStems];
        newInstrumentStems[index] = data.processed_audio;
        setInstrumentStems(newInstrumentStems);
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    }
    
    audioRefs.current.forEach((audio) => audio.currentTime = 0);
    setCurrentTime(0);
  };

  const handleExportPopupOpen = () => {
    setExportPopupOpen(true);
    document.body.classList.add('export-popup-open');
  };

  const handleExportPopupClose = () => {
    setExportPopupOpen(false);
    document.body.classList.remove('export-popup-open');
  };

const handleExport = async () => {
  try {
    // Create FormData object to send files
    const formData = new FormData();

    selectedTracks.forEach(index => {
      const audioRef = audioRefs.current[index];
      
      // Extract the base64 encoded audio data
      const audioData = atob(audioRef.src.split(',')[1]);

      // Convert base64 encoded audio data to binary
      const binaryData = audioData;
      
      // Convert binary string to ArrayBuffer
      const arrayBuffer = new ArrayBuffer(binaryData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      for (let i = 0; i < binaryData.length; i++) {
        uint8Array[i] = binaryData.charCodeAt(i);
      }
      // Create Blob from ArrayBuffer
      const audioBlob = new Blob([arrayBuffer], { type: 'audio/wav' });
      
      // Get the instrument name for the current index
      const instrumentName = instrumentNames[index];
      
      // Append File to FormData with instrument name as filename
      formData.append('audio_files', audioBlob, `${instrumentName}.wav`);
      formData.append('volume_values', audioRef.volume);
      formData.append('panning_values', trackPans[index] / 100);
    });

    const response = await fetch('http://127.0.0.1:8000/api/create_export/', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const data = await response.json();
      console.log('Exported data:', data.processed_audio_zip);
      // Decode the Base64-encoded zip archive
      const zipData = atob(data.processed_audio_zip);
      // Convert the binary zip data to ArrayBuffer
      const arrayBuffer = new ArrayBuffer(zipData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      for (let i = 0; i < zipData.length; i++) {
        uint8Array[i] = zipData.charCodeAt(i);
      }
      // Create a Blob from the ArrayBuffer
      const blob = new Blob([arrayBuffer], { type: 'application/zip' });
      // Allow the user to save the zip file locally
      saveAs(blob, 'modified_stems.zip');
    } else {
      console.error('Error exporting stems:', response.statusText);
    }
  } catch (error) {
    console.error('Error exporting stems:', error);
  }
  handleExportPopupClose();
};

  const handleTrackSelection = (index) => {
    if (selectedTracks.includes(index)) {
      setSelectedTracks(selectedTracks.filter((trackIndex) => trackIndex !== index));
    } else {
      setSelectedTracks([...selectedTracks, index]);
    }
  };

  const updateTrackPan = (index, panValue) => {
    const newPans = [...trackPans];
    newPans[index] = panValue;
    setTrackPans(newPans);
  };


const handleDrop = async (event) => {
  event.preventDefault();
  const files = event.dataTransfer.files;
  // Iterate through dropped files
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const reader = new FileReader();
    reader.onload = async (e) => {
      console.log('Audio file detected!');
      const audioData = e.target.result;
      // Create a new InstrumentTrack for the dropped file
      const newInstrumentStem = audioData.split(',')[1]; // Extract base64 encoded audio data
      const formData = new FormData();
      formData.append('audio_file', file);
      try {
        const response = await fetch('http://127.0.0.1:8000/api/get_image/', {
          method: 'POST',
          body: formData
        });
        if (response.ok) {
          const data = await response.json();
          console.log('Waveform image:', data.waveform_image);
          // Set the new InstrumentTrack's waveform image and audio data
          setWaveformImages([...waveformImages, data.waveform_image]);
          setInstrumentStems([...instrumentStems, newInstrumentStem]);
          // Add new audio index and initialize track pan
          setAudioIndexes([...audioIndexes, instrumentStems.length]);
          setOriginalStems([...originalStems, newInstrumentStem]);
          setTrackPans([...trackPans, 0]);
          // Extract filename without extension and add it to instrumentNames array
          const filenameWithoutExtension = file.name.split('.').slice(0, -1).join('.');
          setInstrumentNames([...instrumentNames, filenameWithoutExtension]);
        } else {
          console.error('Error:', response.statusText);
        }
      } catch (error) {
        console.error('Error:', error);
      }
    };
    reader.readAsDataURL(file);
  }
};

  const handleNewProject = () => {
    // Clear all states
    setInstrumentStems([]);
    setSelectedFile(null);
    setWaveformImages([]);
    setAudioIndexes([]);
    setIsPlaying(false);
    setCurrentTime(0);
    setOriginalStems([]);
    setSelectedTracks([]);
    setTrackPans([]);
    setSelectedEffectsList([]);
    setProjects([]);
    // Ensure all audio players are reset
    if (audioRefs.current.length > 0) {
      audioRefs.current.forEach(audio => {
        audio.pause();
        audio.currentTime = 0;
      });
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Clear the file input
    }
    // Trigger a reset in InstrumentTrack components
    setResetTrigger(true);
  };
  

const handleSaveProject = async () => {
  const projectData = {
    instrumentStems,
    selectedFile,
    waveformImages,
    audioIndexes,
    isPlaying,
    currentTime,
    originalStems,
    selectedTracks,
    trackPans,
    instrumentNames,
    resetTrigger,
    instruments: []
  };

  instrumentStems.forEach((_, index) => {
    const audioRef = audioRefs.current[index];
    const instrument = {
      name: instrumentNames[index],
      volume: audioRef ? audioRef.volume * 100 : 50,
      pan: trackPans[index],
      effects: selectedEffectsList[index] || []
    };
    projectData.instruments.push(instrument);
  });

  const jsonString = JSON.stringify(projectData, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const renamedFile = new File([blob], `${projectName}.json`, { type: 'application/json' });

  const formData = new FormData();
  formData.append('credential', token.credential.slice(0, 255));
  formData.append('project_name', projectName);
  formData.append('file', renamedFile);

  try {
    const response = await fetch('http://127.0.0.1:8000/api/upload_and_add_project/', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const data = await response.json();
      console.log('Project saved and uploaded:', data);
    } else {
      console.error('Error:', response.statusText);
    }
  } catch (error) {
    console.error('Error:', error);
  }
};

const handleDeleteProject = async (projectName) => {
  const credential = token.credential.slice(0, 255);
  const response = await fetch('http://127.0.0.1:8000/api/delete_project/', {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ credential, project_name: projectName })
  });

  if (response.ok) {
    setProjects((prevProjects) => prevProjects.filter((project) => project !== projectName));
    console.log('Project deleted successfully');
  } else {
    console.error('Error deleting project:', response.statusText);
  }
};

const handleLoadProject = async (projectName) => {
  const credential = token.credential.slice(0, 255);
  try {
    const response = await fetch(`http://127.0.0.1:8000/api/get_project_file/?credential=${credential}&project_name=${projectName}`, {
      method: 'GET',
    });

    if (response.ok) {
      const jsonData = await response.json();

      // Initialize variables with values from the loaded project
      setInstrumentStems(jsonData.instrumentStems);
      setSelectedFile(jsonData.selectedFile);
      setWaveformImages(jsonData.waveformImages);
      setAudioIndexes(jsonData.audioIndexes);
      setIsPlaying(jsonData.isPlaying);
      setCurrentTime(jsonData.currentTime);
      setOriginalStems(jsonData.originalStems);
      setSelectedTracks(jsonData.selectedTracks);
      setTrackPans(jsonData.trackPans);
      setInstrumentNames(jsonData.instrumentNames);
      setResetTrigger(jsonData.resetTrigger);

      // Extract and set volumes and pans from instruments
      const volumes = jsonData.instruments.map(instrument => instrument.volume);
      const pans = jsonData.instruments.map(instrument => instrument.pan);
      const effects = jsonData.instruments.map(instrument => instrument.effects);
      setInstrumentVolumes(volumes);
      setInstrumentPans(pans);
      setSelectedEffectsList(effects);

      console.log('Loaded instrument volumes:', volumes);
      console.log('Loaded instrument pans:', pans);
      console.log('Loaded instrument effects:', effects);

      // Initialize variables in InstrumentTrack module
      audioRefs.current = jsonData.instrumentStems.map(() => React.createRef());

      console.log('Project loaded successfully');
    } else {
      console.error('Error loading project file:', response.statusText);
    }
  } catch (error) {
    console.error('Error:', error);
  }
};





// Step 2: Implement drag over event handler to prevent default behavior
const handleDragOver = (event) => {
  event.preventDefault();
};

  return (
    <div>
      <header>
        <h1>AI Instrument Splitter</h1>
        {!isLoggedIn && (
          <GoogleOAuthProvider clientId="114990774632-cq983n39naeea547olvl998p51snb2li.apps.googleusercontent.com">
            <GoogleLogin 
              className="google"
              onSuccess={(token) => {
              setIsLoggedIn(true);
              setToken(token);
              console.log('Token received:', token);
            }} onError={errorMessage} />
          </GoogleOAuthProvider>
        )}
      </header>
      {isLoggedIn && (
      <div className="input-container" onDrop={handleDrop} onDragOver={handleDragOver}>
        <input type="file" ref={fileInputRef} onChange={handleFileChange} />
        <button className="upload-button" onClick={uploadFile}>Upload</button>
        <button className="upload-button" onClick={handleNewProject}>New Project</button>
        <button className="upload-button" onClick={handleSaveProject}>Save Project</button>
      </div>
      )}
      
      {isLoggedIn && (
        <div className="projects-list">
          <h2>Projects</h2>
          <ul>
            {projects.map((project, index) => (
              <li key={index} className="project-item">
                <button onClick={() => handleLoadProject(project)}>{project}</button>
                <button className="delete-button" onClick={() => handleDeleteProject(project)}>
                  <img src={icon} alt="Delete" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {loading && ( // Display animation while waiting for response
        <div className="loading-container">
          <img src={animation} alt="Loading Animation" />
          <p>Hang on, our AI likes to make an entrance</p>
        </div>
      )}
      <div className="player-container">
        {instrumentStems.length > 0 && (
          <>
            <button className="play-button" onClick={togglePlay}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <progress className="progress-bar" max={audioRefs.current.length > 0 && audioRefs.current[0] && !isNaN(audioRefs.current[0].duration) ? audioRefs.current[0].duration : 0} value={currentTime} onClick={handleProgressClick}></progress>
          </>
        )}
      </div>
      <div className="instrument-stems">
        {instrumentStems.length > 0 && instrumentStems.map((instrument, index) => (
          <InstrumentTrack key={index} index={index} instrument={instrument} waveformImage={waveformImages[index]} audioRefs={audioRefs} handleMuteToggle={handleMuteToggle} handleSoloToggle={handleSoloToggle} handleEffectAdd={handleEffectAdd} handleEffectAdd2={handleEffectAdd2} updateTrackPan={updateTrackPan} stemName={instrumentNames[index]} resetTrigger={resetTrigger} setResetTrigger={setResetTrigger} updateSelectedEffects={updateSelectedEffects} loadedVolume={instrumentVolumes[index]} loadedPan={instrumentPans[index]} loadedEffects={selectedEffectsList[index]}/>
        ))}
      </div>
      {audioIndexes.map((index) => (
        <audio
          key={index}
          ref={(element) => audioRefs.current[index] = element}
          src={`data:audio/wav;base64,${instrumentStems[index]}`}
          onTimeUpdate={handleTimeUpdate}
          onEnded={() => setIsPlaying(false)}
        />
      ))}
      
      <div>
        {instrumentStems.length > 0 && (
          <h3 className="drop" onDrop={handleDrop} onDragOver={handleDragOver}>Drop an audio stem here to load it in the project!</h3>
         )}
      </div>
      
      {/* Render export button only if InstrumentTrack components are present */}
      {instrumentStems.length > 0 && (
        <button className="export-button" onClick={handleExportPopupOpen}>
          Export
        </button>
      )}


      {exportPopupOpen && (
        <div className="export-popup">
          <div className="export-popup-content">
            <h2>Select tracks to export</h2>
            <form>
              {instrumentStems.map((_, index) => (
                <div key={index}>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedTracks.includes(index)}
                      onChange={() => handleTrackSelection(index)}
                    />
                    {instrumentNames[index]}
                  </label>
                </div>
              ))}
              <button type="button" onClick={handleExport}>
                Export Selected
              </button>
              <button type="button" onClick={handleExportPopupClose}>
                Cancel
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

const InstrumentTrack = ({ index, instrument, waveformImage, audioRefs, handleMuteToggle, handleSoloToggle, handleEffectAdd, handleEffectAdd2, updateTrackPan, stemName, resetTrigger, setResetTrigger, updateSelectedEffects, loadedVolume, loadedPan, loadedEffects }) => {
  const { volume, pan, effects } = instrument;
  const [selectedEffects, setSelectedEffects] = useState(effects || []);
  const [currentVolume, setCurrentVolume] = useState(volume || 50);
  const [currentPan, setCurrentPan] = useState(pan || 0);
  const panNode = useRef(null);
  
  useEffect(() => {
    if (resetTrigger) {
      setSelectedEffects([]);
      setCurrentVolume(50);
      setCurrentPan(0);
      if (panNode.current) {
        panNode.current.pan.value = 0;
      }
      setResetTrigger(false);
    }
  }, [resetTrigger]);

  useEffect(() => {
    initializeVolume(50);
  }, []);

  useEffect(() => {
    const audioRef = audioRefs.current[index];
    if (audioRef && !panNode.current) {
      const audioContext = new AudioContext();
      panNode.current = audioContext.createStereoPanner();
      const source = audioContext.createMediaElementSource(audioRef);
      source.connect(panNode.current).connect(audioContext.destination);
    }
  }, [audioRefs, index]);

  useEffect(() => {
    updateSelectedEffects(index, selectedEffects);
  }, [selectedEffects]);
  
useEffect(() => {
    const updateVolumeAndPan = () => {
        if (loadedVolume !== undefined && loadedPan !== undefined) {
            handleVolumeChange2(loadedVolume);
            handlePanChange2(loadedPan);
            setSelectedEffects(loadedEffects);
        }
    };

    updateVolumeAndPan(); // Initial check
}, []); // Empty dependency array to run after all other useEffects


  const initializeVolume = (newVolume) => {
    setCurrentVolume(newVolume);
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.volume = newVolume / 100;
    }
  };

  const handleVolumeChange = (event) => {
    const newVolume = parseFloat(event.target.value);
    setCurrentVolume(newVolume);
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.volume = newVolume / 100;
    }
  };

  const handlePanChange = (event) => {
    const newPan = parseInt(event.target.value);
    setCurrentPan(newPan);
    if (panNode.current) {
      const normalizedPan = newPan / 100;
      panNode.current.pan.value = normalizedPan;
      updateTrackPan(index, newPan);
    }
  };
  
  const handleVolumeChange2 = (loadedVolume) => {
    setCurrentVolume(loadedVolume);
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.volume = loadedVolume / 100;
    }
  };

  const handlePanChange2 = (loadedPan) => {
    setCurrentPan(loadedPan);
    if (panNode.current) {
      const normalizedPan = loadedPan / 100;
      panNode.current.pan.value = normalizedPan;
      updateTrackPan(index, loadedPan);
    }
  };


  const handleAddEffect = (selectedEffect) => {
    if (!selectedEffects.includes(selectedEffect)) {
      const newEffects = [...selectedEffects, selectedEffect];
      setSelectedEffects(newEffects);
      handleEffectAdd(index, selectedEffect);
    }
  };

  const handleRemoveEffect = (effectToRemove) => {
    const newEffects = selectedEffects.filter((effect) => effect !== effectToRemove);
    setSelectedEffects(newEffects);
    handleEffectAdd2(index, newEffects);
  };

  return (
    <div className="instrument-track">
      <h3>{stemName}</h3>
      <div className="instrument-track-settings">
        <div>
          Volume: <span>{currentVolume}</span>
          <input type="range" min="0" max="100" value={currentVolume} onChange={handleVolumeChange} />
        </div>
        <div>
          Pan: <span>{currentPan}</span>
          <input type="range" min="-100" max="100" value={currentPan} onChange={handlePanChange} />
        </div>
        <button onClick={() => handleMuteToggle(index)} className="mute-button">
          {audioRefs.current[index] && audioRefs.current[index].muted ? 'Unmute' : 'Mute'}
        </button>
        <button onClick={() => handleSoloToggle(index)} className="mute-button">
          Solo
        </button>
        <div>
          Effects:
          <ul className="effects-list">
            {selectedEffects &&
              selectedEffects.map((effect, i) => (
                <li key={i}>
                  {effect}
                  <button onClick={() => handleRemoveEffect(effect)}>Remove</button>
                </li>
              ))}
          </ul>
          <span>Add Effect:</span>
          <select onChange={(event) => handleAddEffect(event.target.value)}>
            <option value="">Select Effect</option>
            {availableEffects.map((effect, i) => (
              <option key={i} value={effect}>
                {effect}
              </option>
            ))}
          </select>
        </div>
      </div>
      {waveformImage && (
        <div className="waveform-container">
          <img src={`data:image/png;base64,${waveformImage}`} alt="Waveform" />
        </div>
      )}
    </div>
  );
};


export default App;

