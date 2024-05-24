import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import animation from './animation.gif';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';
import { GoogleLogin } from '@react-oauth/google';
import { GoogleOAuthProvider } from '@react-oauth/google';

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
        <input type="file" onChange={handleFileChange} />
        <button className="upload-button" onClick={uploadFile}>Upload</button>
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
            <progress className="progress-bar" max={audioRefs.current.length > 0 && !isNaN(audioRefs.current[0].duration) ? audioRefs.current[0].duration : 0} value={currentTime} onClick={handleProgressClick}></progress>
          </>
        )}
      </div>
      <div className="instrument-stems">
        {instrumentStems.length > 0 && instrumentStems.map((instrument, index) => (
          <InstrumentTrack key={index} index={index} instrument={instrument} waveformImage={waveformImages[index]} audioRefs={audioRefs} handleMuteToggle={handleMuteToggle} handleSoloToggle={handleSoloToggle} handleEffectAdd={handleEffectAdd} handleEffectAdd2={handleEffectAdd2} updateTrackPan={updateTrackPan} stemName={instrumentNames[index]}/>
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

const InstrumentTrack = ({ index, instrument, waveformImage, audioRefs, handleMuteToggle, handleSoloToggle, handleEffectAdd, handleEffectAdd2, updateTrackPan, stemName }) => {
  const instrumentNames = ['Voice', 'Drums', 'Bass', 'Other'];
  const { volume, pan, effects } = instrument;
  const [selectedEffects, setSelectedEffects] = useState(effects || []);
  const [currentVolume, setCurrentVolume] = useState(volume || 50);
  const [currentPan, setCurrentPan] = useState(pan || 0);
  const panNode = useRef(null);

  useEffect(() => {
    // Call handleVolumeChange function to set initial volume
    initializeVolume(50);
  }, []); // Empty dependency array ensures that this effect runs only once, similar to componentDidMount

  useEffect(() => {
    const audioRef = audioRefs.current[index];
    if (audioRef && !panNode.current) {
      // Create a StereoPannerNode for panning
      const audioContext = new AudioContext();
      panNode.current = audioContext.createStereoPanner();
      const source = audioContext.createMediaElementSource(audioRef);
      source.connect(panNode.current).connect(audioContext.destination);
    }
  }, [audioRefs, index]);

  const initializeVolume = (newVolume) => {
    setCurrentVolume(newVolume);
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.volume = newVolume / 100; // Normalize the volume to a range of 0 to 1
    }
  };

  const handleVolumeChange = (event) => {
    const newVolume = parseFloat(event.target.value);
    setCurrentVolume(newVolume);
    const audioRef = audioRefs.current[index];
    if (audioRef) {
      audioRef.volume = newVolume / 100; // Normalize the volume to a range of 0 to 1
    }
  };

  const handlePanChange = (event) => {
    const newPan = parseInt(event.target.value);
    setCurrentPan(newPan);
    if (panNode.current) {
      const normalizedPan = newPan / 100;
      panNode.current.pan.value = normalizedPan;
      updateTrackPan(index, newPan); // Update the track pan in the parent component
    }
  };

  const handleAddEffect = (selectedEffect) => {
    if (!selectedEffects.includes(selectedEffect)) {
      setSelectedEffects([...selectedEffects, selectedEffect]);
      handleEffectAdd(index, selectedEffect); // Call handleEffectAdd to add the effect
    }
  };

  const handleRemoveEffect = (effectToRemove) => {
    setSelectedEffects(selectedEffects.filter((effect) => effect !== effectToRemove));
    // Pass the updated list of effects to handleEffectAdd2
    handleEffectAdd2(index, selectedEffects.filter((effect) => effect !== effectToRemove));
  };

  return (
    <div className="instrument-track">
      <h3>{stemName}</h3> {/* Display the instrument name based on index */}
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
              selectedEffects.map((effect, index) => (
                <li key={index}>
                  {effect}
                  <button onClick={() => handleRemoveEffect(effect)}>Remove</button>
                </li>
              ))}
          </ul>
          <span>Add Effect:</span>
          <select onChange={(event) => handleAddEffect(event.target.value)}>
            <option value="">Select Effect</option>
            {availableEffects.map((effect, index) => (
              <option key={index} value={effect}>
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

