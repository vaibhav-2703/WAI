<h1 id="realtime-face-recognition-system">RealTime Face Recognition System</h1>
<p>A high-accuracy, Python-based face recognition system designed to recognize faces in real time using webcam input. Achieves 92%+ accuracy with efficient algorithms, requiring minimal computational resources and no additional hardware like IR sensors.</p>
<p>🚀 Features</p>
<pre><code>🎥 Real-Time Recognition: Detect and recognize faces live using a webcam.
📊 High Accuracy: Achieves 92%+ recognition accuracy on a standard dataset.
🧠 Efficient Algorithms: Uses MTCNN for face detection and ResNet for embeddings.
💾 Dynamic Scalability: Add new faces dynamically without retraining the entire model.
📷 Camera Selection: Select from multiple available webcams on your device.
🛡️ Customizable Threshold: Fine-tune similarity thresholds for recognition precision.
</code></pre>
<h2 id="🛠️-installation">🛠️ Installation</h2>
<ul>
<li><p>Prerequisites</p>
<ul>
<li>Python 3.7+</li>
<li>Torch (for deep learning models)</li>
<li>OpenCV (for video input)</li>
<li>Facenet-PyTorch (for MTCNN and ResNet)</li>
<li>Scikit-Learn (for SVM classifier)</li>
</ul>
</li>
<li><p>Steps to Set Up</p>
<ul>
<li>Clone the Repository:</li>
</ul>
</li>
</ul>
<pre><code>git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
</code></pre>
<ul>
<li>Install Dependencies:</li>
</ul>
<pre><code>pip install -r requirements.txt
</code></pre>
<ul>
<li>Run the Program:</li>
</ul>
<pre><code>    python face_recognition_system.py
</code></pre>
<h2 id="📋-usage">📋 Usage</h2>
<ol>
<li><p>Train the Model</p>
<ul>
<li>Prepare your dataset in the following structure:</li>
</ul>
</li>
</ol>
<pre><code>dataset/
├── Person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── Person2/
    ├── img1.jpg
    ├── img2.jpg
</code></pre>
<ul>
<li>Start the program:</li>
</ul>
<pre><code>    python face_recognition_system.py
</code></pre>
<ul>
<li>Select &quot;train&quot; when prompted and provide the dataset path.</li>
</ul>
<ol start="2">
<li>Recognize Faces</li>
</ol>
<ul>
<li>Start the program:</li>
</ul>
<pre><code>    python face_recognition_system.py
</code></pre>
<ul>
<li>Select &quot;recognize&quot; when prompted. The system will use your webcam for real-time recognition.</li>
</ul>
<ol start="3">
<li><p>Camera Selection</p>
<ul>
<li>During recognition, choose a specific webcam index if multiple cameras are connected.</li>
</ul>
</li>
</ol>
<h2 id="⚙️-how-it-works">⚙️ How It Works</h2>
<pre><code>Face Detection:
    -Uses MTCNN to detect and align faces from webcam frames or images.

Embedding Extraction:
   - Extracts 512-dimensional face embeddings using ResNet (InceptionResnetV1).

Classification:
   - Trains an SVM classifier to recognize faces based on embeddings.

Real-Time Recognition:
   - Matches detected faces against known faces using cosine similarity and the trained SVM.
</code></pre>
<h2 id="📊-results">📊 Results</h2>
<ul>
<li>Accuracy: Achieved 92%+ accuracy on a custom dataset.</li>
<li>Performance: Optimized for real-time use on devices with limited resources.</li>
</ul>
<h2 id="🧪-testing">🧪 Testing</h2>
<ul>
<li><p>Manual Testing</p>
<ul>
<li>Train the system with a small dataset and verify recognition using the webcam.</li>
</ul>
</li>
<li><p>Automated Testing</p>
<ul>
<li>Add unit tests for embedding extraction, classification, and webcam input handling.</li>
</ul>
</li>
</ul>
<h2 id="📂-repository-structure">📂 Repository Structure</h2>
<pre><code>face-recognition-system/
├── face_recognition_system.py  # Combined main script
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
├── README.md                   # Project documentation
└── dataset/                    # Example dataset (user-provided)
</code></pre>
<h2 id="📜-license">📜 License</h2>
<p>This project is licensed under the MIT License. See the LICENSE file for more details.</p>
<h2 id="🙏-acknowledgments">🙏 Acknowledgments</h2>
<pre><code>Facenet-PyTorch for face detection and embedding extraction.
Scikit-Learn for SVM classifier implementation.
</code></pre>
<h2 id="🌟-future-enhancements">🌟 Future Enhancements</h2>
<pre><code>Integrate face anti-spoofing for enhanced security.
Add REST API for external integration.
Support large-scale datasets with distributed training.
</code></pre>
