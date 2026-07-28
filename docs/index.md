This website provides 3d visualization and processing tools for *Drosophila melanogaster* pose data. 
<style>
  .article-grid {
    grid-template-columns: minmax(0, 1fr) !important;
  }

  main {
    max-width: none !important;
    width: 100% !important;
  }

  .content {
    max-width: none !important;
    width: 100% !important;
  }

  .streamlit-wrapper {
    width: calc(100vw - 60px);
    margin-left: calc(50% - 50vw + 30px);
    height: 1400px;
  }

  .streamlit-wrapper iframe {
    width: 100%;
    height: 100%;
    border: 0;
  }
</style>
<div class="streamlit-wrapper">
  <iframe
    src="https://seqikpy-3d-kinematics-explorer-9uskckjvshmairnfrbluts.streamlit.app/?embed=true"
    title="SeqIKPy 3D Kinematics Explorer"
    loading="lazy"
  ></iframe>
</div>
