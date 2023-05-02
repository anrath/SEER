# Similarity analysis
echo "Executing a similarity analysis..."
python real_data_gen/project_similarity_analysis.py

# Latex Files
echo "Generating LaTeX tables..."
cd real_data_gen/
python latex.py
