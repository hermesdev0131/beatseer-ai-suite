import streamlit as st
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import all modules
from config import config, get_project_dirs
from audio_processor import AudioProcessor
from ml_predictor import MLPredictor
from report_generator import ReportGenerator
from music_intelligence_suite import MusicIntelligenceSuite

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Beatseer AI Hit and Commercial Prospects Suite - Upload & Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check for URL parameters from landing page (if using integrated flow)
query_params = st.experimental_get_query_params()
if 'type' in query_params:
    analysis_type = query_params['type'][0]  # experimental_get_query_params returns lists
    if analysis_type in ['album', 'batch']:
        st.session_state.analysis_type = analysis_type
        # Clear the parameter after reading (optional in older versions)
        st.experimental_set_query_params()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .suite-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #ff00ff 100%);
        -webkit-background-clip: text;
        color: #00ff88;
    }
    .suite-subtitle {
        font-size: 1.3rem;
        color: #00ff88;
        margin: 1rem 0;
    }
    .beatseer-badge {
        display: inline-block;
        background: rgba(0, 255, 136, 0.2);
        border: 2px solid #00ff88;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        color: #00ff88;
        font-weight: bold;
        margin-top: 1rem;
    }
    .option-card {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .option-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    .executive-narrative {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,212,255,0.1) 100%);
        border: 2px solid #00ff88;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class MusicIntelligenceApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize all components"""
        # Get project directories
        self.dirs = get_project_dirs()
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=config.SAMPLE_RATE)
        self.ml_predictor = MLPredictor(self.dirs['models'] / 'trained_models.pkl')
        self.report_generator = ReportGenerator(config)
        
        # Initialize Music Intelligence Suite
        self.intelligence_suite = MusicIntelligenceSuite(
            audio_processor=self.audio_processor,
            ml_predictor=self.ml_predictor,
            config=config
        )
        
        # Initialize session state
        self._init_session_state()
        
        logger.info("Music Intelligence App initialized successfully")
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'user_type' not in st.session_state:
            st.session_state.user_type = None
        if 'role' not in st.session_state:
            st.session_state.role = 'artist'
        if 'show_waveforms' not in st.session_state:
            st.session_state.show_waveforms = True
    
    def run(self):
        """Main application entry point"""
        # Display header
        self._show_header()
        
        # Check if analysis is complete
        if st.session_state.get('analysis_complete'):
            self._show_results()
        else:
            self._show_analysis_selection()

    
    def _show_header(self):
        """Display application header"""
        st.markdown("""
        <div class="main-header">
            <h1 class="suite-title">Music Intelligence Suite</h1>
            <div class="suite-subtitle">Hit Prediction • Sync Licensing • Investment Analysis</div>
            <div class="beatseer-badge">Powered by Beatseer AI™</div>
        </div>
        """, unsafe_allow_html=True)

    


    def _show_analysis_selection(self):
        """Show main analysis type selection"""
        
        st.markdown("## 🎯 Select Your Analysis Type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="option-card">
                <h3>💿 Album/EP Analysis</h3>
                <p><strong>Singles Selection • Release Strategy • Cohesion Analysis</strong></p>
                <ul>
                    <li>Identify top 3 singles</li>
                    <li>Album flow analysis</li>
                    <li>Strategic release timeline</li>
                    <li>Marketing recommendations</li>
                </ul>
                <p><em>Ideal for: Artists, Producers, Independent Labels</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start Album Analysis", key="album_btn", use_container_width=True):
                st.session_state.analysis_type = 'album'
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="option-card">
                <h3>🎯 Professional Batch Analysis</h3>
                <p><strong>A&R Decisions • Sync Potential • Investment Intelligence</strong></p>
                <ul>
                    <li>Signing recommendations</li>
                    <li>Sync licensing valuation</li>
                    <li>Investment risk assessment</li>
                    <li>Executive AI narrative</li>
                </ul>
                <p><em>Ideal for: Labels, Music Supervisors, Publishers</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start Batch Analysis", key="batch_btn", use_container_width=True):
                st.session_state.analysis_type = 'batch'
                st.rerun()
        
        # Show selected interface
        if st.session_state.get('analysis_type'):
            st.markdown("---")
            if st.session_state.analysis_type == 'album':
                self._show_album_interface()
            else:
                self._show_batch_interface()

    def _show_album_interface(self):
        """Album analysis interface"""
        st.markdown("### 💿 Album/EP Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            album_title = st.text_input("Album/EP Title:", placeholder="Enter album title")
        with col2:
            artist_name = st.text_input("Artist Name:", placeholder="Enter artist name")
        
        # Role selection
        st.markdown("#### Select Analysis Perspective")
        role_col1, role_col2, role_col3 = st.columns(3)
        
        with role_col1:
            if st.button("🎤 Artist", use_container_width=True):
                st.session_state.role = 'artist'
        with role_col2:
            if st.button("🎛️ Producer", use_container_width=True):
                st.session_state.role = 'producer'
        with role_col3:
            if st.button("💼 Label", use_container_width=True):
                st.session_state.role = 'label'
        
        if st.session_state.get('role'):
            st.success(f"Analysis perspective: {st.session_state.role.title()}")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Album Tracks (3-15 files)",
            type=['mp3', 'wav', 'flac', 'm4a', 'ogg'],
            accept_multiple_files=True,
            help="Name files as 'Artist - Track.mp3' for best results"
        )
        
        if uploaded_files:
            if len(uploaded_files) < 3:
                st.warning("⚠️ Please upload at least 3 tracks")
            elif len(uploaded_files) > 15:
                st.warning("⚠️ Maximum 15 tracks allowed")
            else:
                st.success(f"✅ {len(uploaded_files)} tracks ready")
                # Show track list
                with st.expander("📋 View Album Tracks"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.text(f"{i}. {file.name} ({file.size / (1024*1024):.2f} MB)")
                
                if album_title:
                    if st.button("🚀 Analyze Album", use_container_width=True):
                        with st.spinner("Analyzing with Beatseer AI..."):
                            results = self.intelligence_suite.analyze_album(
                                uploaded_files,
                                st.session_state.get('role', 'artist'),
                                album_title,
                                artist_name
                            )
                            st.session_state.results = results
                            st.session_state.analysis_complete = True
                            st.rerun()
                else:
                    st.warning("⚠️ Please enter an album title")


    def _show_batch_interface(self):
        """Professional batch analysis interface"""
        st.markdown("### 🎯 Professional Batch Analysis")
        
        # User type selection
        st.markdown("#### Select Your Role")
        user_col1, user_col2, user_col3 = st.columns(3)
        
        with user_col1:
            if st.button("🏢 Record Label", use_container_width=True):
                st.session_state.user_type = 'label'
        with user_col2:
            if st.button("🎬 Music Supervisor", use_container_width=True):
                st.session_state.user_type = 'supervisor'
        with user_col3:
            if st.button("📈 Publisher", use_container_width=True):
                st.session_state.user_type = 'publisher'
        
        if st.session_state.get('user_type'):
            role_names = {
                'label': 'A&R Decision Intelligence',
                'supervisor': 'Sync Licensing Intelligence',
                'publisher': 'Portfolio Intelligence'
            }
            st.success(f"Mode: {role_names[st.session_state.user_type]}")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_name = st.text_input(
                "Batch Name:",
                value=f"Analysis - {datetime.now().strftime('%B %Y')}"
            )
        with col2:
            company_name = st.text_input("Organization:", placeholder="Your company")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Submissions (3-15 tracks)",
            type=['mp3', 'wav', 'flac', 'm4a', 'ogg'],
            accept_multiple_files=True,
            help="Format: 'Artist - Track.mp3' for best results"
        )
        
        if uploaded_files:
            if len(uploaded_files) < 3:
                st.warning("⚠️ Please upload at least 3 tracks")
            elif len(uploaded_files) > 15:
                st.warning("⚠️ Maximum 15 tracks allowed")
            else:
                st.success(f"✅ {len(uploaded_files)} submissions ready")
                # Show submission list
                with st.expander("📋 View Submissions"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.text(f"{i}. {file.name} ({file.size / (1024*1024):.2f} MB)")
                
                if st.session_state.get('user_type'):
                    if st.button("🚀 Generate Intelligence Report", use_container_width=True):
                        with st.spinner("Analyzing with Beatseer AI..."):
                            results = self.intelligence_suite.analyze_professional_batch(
                                uploaded_files,
                                st.session_state.get('user_type', 'label'),
                                batch_name,
                                company_name
                            )
                            st.session_state.results = results
                            st.session_state.analysis_complete = True
                            st.rerun()
                else:
                    st.warning("⚠️ Please select your role")



    
    def _show_results(self):
        """Display results with executive narrative"""
        results = st.session_state.results
        
        # Title based on analysis type
        if results['analysis_type'] == 'album':
            st.markdown(f"# 💿 Album Intelligence Report: {results['album_title']}")
        else:
            st.markdown(f"# 🎯 Professional Intelligence Report: {results['batch_name']}")
        
        # Executive Summary Metrics
        st.markdown("## Executive Summary")
        
        if results['analysis_type'] == 'album':
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = results['album_metrics']
            with col1:
                st.metric("Avg Hit %", f"{metrics['average_hit_probability']:.0f}%")
            with col2:
                st.metric("Cohesion", f"{results['album_analysis']['overall_cohesion']:.0f}%")
            with col3:
                st.metric("Strong Tracks", metrics['strong_tracks'])
            with col4:
                st.metric("Genre", results['album_analysis']['primary_genre'].title())
            with col5:
                st.metric("Rating", metrics['album_strength'])
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = results['batch_metrics']
            with col1:
                st.metric("Avg Hit %", f"{metrics['average_hit_potential']:.0f}%")
            with col2:
                st.metric("Signable", metrics['signable_tracks'])
            with col3:
                st.metric("Portfolio", metrics['total_portfolio_value'])
            with col4:
                st.metric("ROI", metrics['roi_projection'])
            with col5:
                st.metric("Sync Ready", metrics['production_ready'])
        
        # Executive AI Narrative
        st.markdown("## 🎯 Beatseer AI Executive Intelligence Brief")
        st.markdown('<div class="executive-narrative">', unsafe_allow_html=True)
        
        # Display formatted narrative
        narrative = results.get('executive_narrative', '')
        narrative_sections = narrative.split('\n\n')
        
        for section in narrative_sections:
            if section.strip():
                lines = section.split('\n')
                for line in lines:
                    if line.startswith('===') or line.startswith('---'):
                        continue
                    elif any(header in line for header in ['EXECUTIVE', 'COMMERCIAL', 'STRATEGIC', 'RISK', 'INVESTMENT', 'FINAL']):
                        st.markdown(f"### {line.strip()}")
                    elif line.strip():
                        st.markdown(line.strip())
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Analysis Tabs
        if results['analysis_type'] == 'album':
            tabs = st.tabs(["🎯 Singles", "📊 Tracks", "📈 Strategy", "💾 Export"])
            
            with tabs[0]:
                self._show_singles_tab(results)
            with tabs[1]:
                self._show_tracks_tab(results)
            with tabs[2]:
                self._show_strategy_tab(results)
            with tabs[3]:
                self._show_export_tab(results)
        else:
            tabs = st.tabs(["🎯 Priorities", "📊 Analysis", "🎬 Sync", "📈 Market", "💾 Export"])
            
            with tabs[0]:
                self._show_priorities_tab(results)
            with tabs[1]:
                self._show_analysis_tab(results)
            with tabs[2]:
                self._show_sync_tab(results)
            with tabs[3]:
                self._show_marketing_tab(results)
            with tabs[4]:
                self._show_export_options(results)
        
        # New Analysis Button
        if st.button("🔄 Start New Analysis", use_container_width=True):
            for key in ['analysis_complete', 'results', 'analysis_type', 'role', 'user_type']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    def _show_singles_tab(self, results):
        """Show singles recommendations"""
        singles = results['singles']
        
        if singles['lead_single']:
            st.markdown("### 🥇 Lead Single")
            st.success(f"**{singles['lead_single']['track']}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hit Potential", f"{singles['lead_single']['hit_probability']:.0f}%")
            with col2:
                st.metric("Viral Score", f"{singles['lead_single']['viral_score']:.0f}%")
            with col3:
                st.metric("Track #", singles['lead_single']['track_number'])
            st.caption(singles['lead_single']['why'])
        
        if singles['second_single']:
            st.markdown("### 🥈 Second Single")
            st.info(f"**{singles['second_single']['track']}**")
            st.metric("Hit Potential", f"{singles['second_single']['hit_probability']:.0f}%")
        
        if singles['third_single']:
            st.markdown("### 🥉 Third Single")
            st.warning(f"**{singles['third_single']['track']}**")
            st.metric("Hit Potential", f"{singles['third_single']['hit_probability']:.0f}%")
    
    def _show_tracks_tab(self, results):
        """Show track analysis"""
        track_data = []
        for track in results['track_results']:
            track_data.append({
                'Track #': track['track_number'],
                'Title': track['title'],
                'Hit %': f"{track['hit_probability']:.0f}%",
                'Genre': track['genre'].title(),
                'Energy': f"{track['energy']:.0f}%",
                'Tempo': f"{track['tempo']:.0f} BPM"
            })
        
        df = pd.DataFrame(track_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _show_strategy_tab(self, results):
        """Show release strategy"""
        strategy = results['release_strategy']
        
        st.markdown("### 📅 Release Timeline")
        for item in strategy['timeline']:
            if item['week'] < 0:
                timeframe = f"**{abs(item['week'])} weeks before release**"
            elif item['week'] == 0:
                timeframe = "**Release Day**"
            else:
                timeframe = f"**{item['week']} weeks after release**"
            
            st.markdown(f"{timeframe}: {item['action']}")
            st.caption(item['goal'])
    
    def _show_priorities_tab(self, results):
        """Show priority recommendations"""
        if 'signing_recommendations' in results.get('analysis_focus', {}):
            recs = results['analysis_focus']['signing_recommendations']
            
            st.markdown("### 🚀 Priority Signings")
            for track in recs['priority_signings'][:3]:
                st.success(f"**{track['artist']} - {track['title']}** ({track['hit_probability']:.0f}%)")
                st.write(f"Advance: {track['advance_range']} | Risk: {track['investment_risk']}")
    
    def _show_analysis_tab(self, results):
        """Show full analysis"""
        track_data = []
        for track in results['track_results'][:10]:
            track_data.append({
                'Artist': track['artist'],
                'Track': track['title'],
                'Hit %': f"{track['hit_probability']:.0f}%",
                'Genre': track['genre'].title(),
                'Market Fit': track['market_fit'],
                'Risk': track['investment_risk']
            })
        
        df = pd.DataFrame(track_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _show_sync_tab(self, results):
        """Show sync opportunities"""
        st.markdown("### 🎬 Sync Licensing Opportunities")
        
        for track in results['track_results'][:5]:
            if max(track['sync_potential'].values()) > 70:
                st.markdown(f"**{track['artist']} - {track['title']}**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Film/TV", f"{track['sync_potential']['Film/TV']}%")
                with col2:
                    st.metric("Commercials", f"{track['sync_potential']['Commercials']}%")
                with col3:
                    st.metric("Games", f"{track['sync_potential']['Games']}%")
                with col4:
                    st.metric("Sports", f"{track['sync_potential']['Sports']}%")
                
                st.markdown("---")
    
    def _show_export_options(self, results):
        """Export options"""
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate HTML report
            html_report = self.report_generator._generate_html_report(results)
            st.download_button(
                "📄 Download HTML Report",
                html_report,
                f"beatseer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
                use_container_width=True
            )
        
        with col2:
            # Export JSON
            import json
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📊 Download JSON Data",
                json_data,
                f"beatseer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )

    def _show_marketing_tab(self, results):
        """Show market analysis and trends"""
        st.markdown("### 📈 Market Intelligence & Positioning")
        
        # Market Overview Metrics
        st.markdown("#### Market Performance Overview")
        
        if 'competitive_analysis' in results:
            comp_analysis = results['competitive_analysis']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Batch vs Industry", 
                    f"{comp_analysis['batch_average']:.1f}%",
                    f"{comp_analysis['batch_average'] - comp_analysis['industry_average']:.1f}% vs industry"
                )
            with col2:
                st.metric("Market Percentile", f"{comp_analysis['percentile']}th")
            with col3:
                st.metric("Industry Benchmark", f"{comp_analysis['industry_average']}%")
        
        # Market Gaps Analysis
        if 'market_gaps' in results and results['market_gaps']:
            st.markdown("#### 🎯 Market Opportunities")
            st.info("**Identified Growth Opportunities**")
            
            for gap in results['market_gaps']:
                with st.expander(f"🚀 {gap['opportunity']}"):
                    st.write(f"**Market Insight:** {gap['reason']}")
                    
                    # Add mock market data for demonstration
                    if 'latin' in gap['opportunity'].lower():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Market Growth", "25% YoY")
                            st.metric("Streaming Share", "18.5%")
                        with col2:
                            st.metric("Revenue Potential", "$2.1B")
                            st.metric("Competition Level", "Medium")
                    elif 'chill' in gap['opportunity'].lower() or 'lo-fi' in gap['opportunity'].lower():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Playlist Reach", "45M+")
                            st.metric("Study Market", "$890M")
                        with col2:
                            st.metric("Avg. Streams", "2.3M")
                            st.metric("Competition Level", "Low")
        
        # Genre Market Analysis
        st.markdown("#### 🎵 Genre Market Positioning")
        
        # Analyze genre distribution in current batch
        if 'track_results' in results:
            genre_data = {}
            market_fit_data = {}
            
            for track in results['track_results']:
                genre = track.get('genre', 'unknown').title()
                market_fit = track.get('market_fit', 'Unknown')
                
                if genre in genre_data:
                    genre_data[genre] += 1
                else:
                    genre_data[genre] = 1
                
                if market_fit in market_fit_data:
                    market_fit_data[market_fit] += 1
                else:
                    market_fit_data[market_fit] = 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Genre Distribution**")
                for genre, count in genre_data.items():
                    percentage = (count / len(results['track_results'])) * 100
                    st.progress(percentage / 100, text=f"{genre}: {count} tracks ({percentage:.1f}%)")
            
            with col2:
                st.markdown("**Market Fit Analysis**")
                for fit_level, count in market_fit_data.items():
                    percentage = (count / len(results['track_results'])) * 100
                    if fit_level == 'Strong':
                        st.success(f"✅ {fit_level}: {count} tracks ({percentage:.1f}%)")
                    elif fit_level == 'Moderate':
                        st.warning(f"⚠️ {fit_level}: {count} tracks ({percentage:.1f}%)")
                    else:
                        st.error(f"❌ {fit_level}: {count} tracks ({percentage:.1f}%)")
        
        # Viral Potential & Market Trends
        st.markdown("#### 🔥 Viral Potential & Trends")
        
        if 'track_results' in results:
            viral_tracks = [t for t in results['track_results'] if t.get('viral_score', 0) > 70]
            
            if viral_tracks:
                st.success(f"🚀 {len(viral_tracks)} tracks with high viral potential identified")
                
                for track in viral_tracks[:3]:  # Show top 3
                    with st.expander(f"🎵 {track.get('artist', 'Unknown')} - {track.get('title', 'Unknown')}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Viral Score", f"{track.get('viral_score', 0):.0f}%")
                        with col2:
                            st.metric("Hit Probability", f"{track.get('hit_probability', 0):.0f}%")
                        with col3:
                            st.metric("Market Fit", track.get('market_fit', 'Unknown'))
                        
                        # Mock social media potential
                        st.markdown("**Social Media Potential:**")
                        st.write("• TikTok: High potential for dance/trend creation")
                        st.write("• Instagram Reels: Strong hook in first 15 seconds")
                        st.write("• YouTube Shorts: Memorable chorus for user-generated content")
            else:
                st.warning("No tracks with high viral potential (>70%) identified in current batch")
        
        # Market Recommendations
        st.markdown("#### 💡 Strategic Market Recommendations")
        
        recommendations = []
        
        if 'batch_metrics' in results:
            metrics = results['batch_metrics']
            avg_hit = metrics.get('average_hit_potential', 0)
            
            if avg_hit > 70:
                recommendations.append("🎯 **Premium Market Positioning**: High hit potential supports premium marketing investment")
                recommendations.append("📺 **Multi-Platform Campaign**: Leverage across streaming, radio, and social media")
            elif avg_hit > 50:
                recommendations.append("🎵 **Targeted Genre Marketing**: Focus on genre-specific playlists and communities")
                recommendations.append("📱 **Social Media First**: Prioritize TikTok and Instagram for organic growth")
            else:
                recommendations.append("🔍 **Niche Market Focus**: Target specific demographics and underground scenes")
                recommendations.append("💰 **Cost-Effective Campaigns**: Emphasize organic growth and word-of-mouth")
        
        # Add market gap recommendations
        if 'market_gaps' in results and results['market_gaps']:
            recommendations.append("🚀 **Market Gap Strategy**: Consider developing content for identified opportunities")
            recommendations.append("🌍 **Diversification**: Expand into underrepresented genres for competitive advantage")
        
        for rec in recommendations:
            st.write(rec)
        
        # Market Timeline
        st.markdown("#### 📅 Market Entry Timeline")
        
        timeline_data = {
            "Phase 1 (Weeks 1-4)": "Market research, audience identification, content creation",
            "Phase 2 (Weeks 5-8)": "Soft launch, influencer partnerships, playlist pitching",
            "Phase 3 (Weeks 9-12)": "Full campaign launch, radio promotion, PR push",
            "Phase 4 (Weeks 13-16)": "Performance optimization, international expansion"
        }
        
        for phase, description in timeline_data.items():
            st.write(f"**{phase}:** {description}")

    def _show_export_tab(self, results):
        """Show export options"""
        col1, col2 = st.columns(2)
        
        with col1:
            # HTML Report
            html_report = self.report_generator._generate_html_report(results)
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name=f"beatseer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col2:
            # JSON Data
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📊 Download JSON Data",
                data=json_data,
                file_name=f"beatseer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    def _process_batch(self, files, batch_name, company_name):
        """Process batch analysis"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🎵 Analyzing submissions...")
            progress_bar.progress(30)
            
            # Run analysis
            results = self.intelligence_suite.analyze_professional_batch(
                files=files,
                user_type=st.session_state.user_type,
                batch_name=batch_name,
                company_name=company_name or ""
            )
            
            status_text.text("🤖 Generating intelligence brief...")
            progress_bar.progress(80)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.session_state.results = results
            st.session_state.analysis_complete = True
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            st.error(f"❌ Analysis failed: {str(e)}")

def main():
    """Main entry point"""
    app = MusicIntelligenceApp()
    app.run()

if __name__ == "__main__":
    main()





