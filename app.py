import io
import re
import streamlit as st
import numpy as np
import requests
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser, CifWriter
from scipy.optimize import linear_sum_assignment
import plotly.graph_objs as go

from aflow import search, K
from aflow import search
import aflow.keywords as AFLOW_K
from pymatgen.core import Structure
import aflow

st.set_page_config(
    page_title="XRDlicious: ",
    layout="wide"
)
# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

# Inject custom CSS for buttons.
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0099ff;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 0.5em 1em;
        border: none;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    div.stButton > button:active,
    div.stButton > button:focus {
        background-color: #0099ff !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stDataFrameContainer"] table td {
         font-size: 22px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <head>
        <meta name="description" content="🍕 XRDlicious – Peak Matcher: Match Experimental Powder XRD Patterns with Structures from Materials Project or AFLOW Databases">
    </head>
    """,
    height=0,
)

st.sidebar.markdown("## 🍕 XRDlicious – Peak Matcher")

col1, col2 = st.columns([1.25, 1])
with col1:
    st.title(
        "🍕 XRDlicious – Peak Matcher: Match Experimental Powder XRD Patterns with Structures from Materials Project or AFLOW Databases")
    st.info(
        "🔬 [Main 🍕 XRDlicious HERE](https://xrdlicious.com/). 🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://youtu.be/ZiRbcgS_cd0)"
    )
from PIL import Image
with col2:
    image = Image.open("images/4.png")
    st.image(image)


convert_to_conventional = True
pymatgen_prim_cell_lll = False
pymatgen_prim_cell_no_reduce = False
MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"


# --- Helper Functions ---


def get_cod_entries(params):
    response = requests.get('https://www.crystallography.net/cod/result', params=params)
    if response.status_code == 200:
        results = response.json()
        return results  # Returns a list of entries
    else:
        st.error(f"COD search error: {response.status_code}")
        return []

def get_cif_from_cod(entry):
    file_url = entry.get('file')
    if file_url:
        response = requests.get(f"https://www.crystallography.net/cod/{file_url}.cif")
        if response.status_code == 200:
            return response.text
    return None

def get_cod_str(cif_content):
    parser = CifParser.from_str(cif_content)
    structure = parser.get_structures(primitive=False)[0]
    return structure

def update_candidate_index():
    try:
        selected_rank = int(st.session_state.selected_candidate.split('.')[0])
        st.session_state.candidate_index = selected_rank - 1
        print("SELECTED")
        print(selected_rank)
    except Exception as e:
        st.session_state.candidate_index = 0


def two_theta_to_d(two_theta, wavelength):
    theta_rad = np.radians(np.array(two_theta) / 2)
    d_spacing = np.where(theta_rad == 0, np.inf, wavelength / (2 * np.sin(theta_rad)))
    return d_spacing


def get_full_conventional_structure(structure, symprec=0.1):
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return analyzer.get_conventional_standard_structure()

import spglib
def get_full_conventional_structure(structure, symprec=1e-3):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords


    species_list = [site.species for site in structure]
    species_to_type = {}
    type_to_species = {}
    type_index = 1

    types = []
    for sp in species_list:
        sp_tuple = tuple(sorted(sp.items()))  # make it hashable
        if sp_tuple not in species_to_type:
            species_to_type[sp_tuple] = type_index
            type_to_species[type_index] = sp
            type_index += 1
        types.append(species_to_type[sp_tuple])

    cell = (lattice, positions, types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    new_species_list = [type_to_species[t] for t in std_types]

    conv_structure = Structure(
        lattice=std_lattice,
        species=new_species_list,
        coords=std_positions,
        coords_are_cartesian=False
    )

    return conv_structure



def get_peak_match_score(exp_peaks, calc_pattern, wavelength, min_intensity=5):
    calc_intensities = np.array(calc_pattern.y)
    valid_indices = np.where(calc_intensities >= min_intensity)[0]
    if len(valid_indices) == 0:
        return 1000 * len(exp_peaks)
    calc_d_spacings = np.array(calc_pattern.d_hkls)[valid_indices]
    exp_d_spacings = two_theta_to_d(exp_peaks, wavelength)
    total_diff = 0
    for exp_d in sorted(exp_d_spacings):
        closest_diff = np.min(np.abs(calc_d_spacings - exp_d))
        total_diff += closest_diff
    return total_diff


def get_peak_match_score_with_intensity(exp_peaks, exp_intensities, calc_pattern, wavelength,
                                        w_angle=1, w_intensity=1, min_intensity=5):
    calc_intensities = np.array(calc_pattern.y)
    valid_indices = np.where(calc_intensities >= min_intensity)[0]
    if len(valid_indices) == 0:
        return 1000 * len(exp_peaks)
    calc_d_spacings = np.array(calc_pattern.d_hkls)[valid_indices]
    exp_d_spacings = two_theta_to_d(exp_peaks, wavelength)
    total_score = 0
    for exp_d, exp_intensity in zip(exp_d_spacings, exp_intensities):
        idx = np.argmin(np.abs(calc_d_spacings - exp_d))
        d_diff = abs(calc_d_spacings[idx] - exp_d)
        intensity_diff = abs(calc_intensities[valid_indices][idx] - exp_intensity)
        total_score += w_angle * d_diff + w_intensity * intensity_diff
    return total_score


full_range = (0.01, 179.9)


def calculate_xrd_pattern(structure, wavelength=1.7809, range=full_range):
    xrd_calc = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calc.get_pattern(structure, two_theta_range=range)
    return pattern


session = requests.Session()


@st.cache_data(show_spinner=False)
def cached_get_structure_from_aflow(auid, aurl, files):
    try:
        # Look for a suitable CIF file.
        cif_files = [f for f in files if f.endswith("_sprim.cif") or f.endswith(".cif")]
        if not cif_files:
            st.warning(f"No CIF files found for AFLOW entry {auid}")
            return None
        cif_filename = cif_files[0]
        # Correct the URL if needed.
        if ":" in aurl:
            host_part, path_part = aurl.split(":", 1)
            corrected_aurl = f"{host_part}/{path_part}"
        else:
            corrected_aurl = aurl
        file_url = f"http://{corrected_aurl}/{cif_filename}"
        response = session.get(file_url)
        if response.status_code == 200:
            cif_content = response.content.decode("utf-8")
            structure = Structure.from_str(cif_content, fmt="cif")
            return structure
        else:
            st.error(f"Failed to fetch CIF from {file_url} (Error: {response.status_code})")
    except Exception as e:
        st.error(f"Error retrieving structure for AFLOW entry {auid}: {e}")
    return None


def get_structure_from_aflow(entry):
    return cached_get_structure_from_aflow(entry.auid, entry.aurl, entry.files)


# --- Streamlit Interface ---


# --- Experimental XRD Data Inputs ---
st.subheader("Experimental XRD Data Inputs")

st.sidebar.header("📤 Upload Experimental XRD Data")
exp_xrd_file = st.sidebar.file_uploader("Upload your XRD file (2 columns: 2θ and intensity)", type=['txt', 'csv', 'xy'])

uploaded_x = uploaded_y = None

if exp_xrd_file is not None:
    try:
        # Load data assuming whitespace or comma delimiter
        user_xrd_data = np.loadtxt(exp_xrd_file, delimiter=None)
        uploaded_x, uploaded_y = user_xrd_data[:, 0], user_xrd_data[:, 1]

        # Normalize intensity
        uploaded_y = 100 * uploaded_y / np.max(uploaded_y)

        st.sidebar.success("✅ XRD file uploaded and normalized successfully.")

        # Preview uploaded data
        if st.sidebar.checkbox("Show uploaded data preview"):
            st.sidebar.write(user_xrd_data[:10, :])
    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {e}")

col00, col01, col02, col03 = st.columns([2.5, 2, 2, 1])
with col00:
    st.markdown(
        "### 🚧 Upload Your XRD data and Automatically find peak positions (Under reconstruction, will be added in future update 🚧)",
        unsafe_allow_html=True)
with col01:
    st.markdown("### Peak Positions (2θ)")
    exp_peaks_str = st.text_input("Enter experimental peak positions separated by commas", "38, 55, 68, 82, 153, 174")
with col02:
    st.markdown("### Intensities (normalized)")
    exp_intensities_str = st.text_input("Enter corresponding experimental intensities separated by commas",
                                        "92, 40, 35, 55, 58, 64")
with col03:
    st.markdown("### Wavelength (Å)")
    user_wavelength = st.number_input("Enter the X-rays wavelength (in Å)", value=1.7809, format="%.4f")

try:
    experimental_peaks = [float(x.strip()) for x in exp_peaks_str.split(",") if x.strip()]
    experimental_intensities = [float(x.strip()) for x in exp_intensities_str.split(",") if x.strip()]
    if len(experimental_peaks) != len(experimental_intensities):
        st.error("The number of peak positions and intensities do not match. Please revise your inputs.")
        experimental_peaks, experimental_intensities = None, None
except Exception as e:
    st.error(f"Error parsing experimental data: {e}")
    experimental_peaks, experimental_intensities = None, None

st.markdown("### Search for Structures in Materials Project or AFLOW Databases")

col1, col2, col3 = st.columns(3)
with col1:
    mp_search_query = st.text_input("Materials Project: Enter elements separated by spaces (e.g., Sr Ti O):",
                                    value="Sr Ti O", key="mp_search")
with col2:
    aflow_elements_input = st.text_input("AFLOW: Enter elements separated by spaces (e.g., Ti O):", value="Ti O",
                                         key="aflow_search")
with col3:
    cod_elements_input = st.text_input("COD: Enter elements separated by spaces (e.g., Ti O)", value="Ti O", key="cod_search")


col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
with col_btn1:
    search_both = st.button("Search All Databases")
with col_btn2:
    search_mp = st.button("Search MP Only")
with col_btn3:
    search_aflow = st.button("Search AFLOW Only")
with col_btn4:
    search_cod = st.button("Search COD Only")


def clear_candidates():
    st.session_state.pop("candidate_list", None)
    st.session_state.pop("candidate_index", None)


# --- Search Section ---
combined_structures = {}

if search_cod:
    clear_candidates()
    with st.spinner(
            f"Searching **the COD database**, please wait. Note that especially the first search typically takes a bit longer while establishing initial connections. 😊"):
        # Build COD search parameters. Here we assume the user enters one or more elements.
        elements = [el.strip() for el in cod_elements_input.split() if el.strip()]
        if elements:
            params = {'format': 'json'}
            # Set search parameters for each element (COD expects keys like 'el1', 'el2', etc.)
            for i, el in enumerate(elements, start=1):
                params[f'el{i}'] = el
            # Set strict search to ensure the number of elements matches exactly.
            params['strictmin'] = str(len(elements))
            params['strictmax'] = str(len(elements))

            cod_entries = get_cod_entries(params)
            if cod_entries:
                status_placeholder = st.empty()
                for entry in cod_entries:
                    cif_content = get_cif_from_cod(entry)
                    if cif_content:
                        try:
                            # Use a unique key (e.g., prefix with "cod_") to avoid collision with MP/AFLOW IDs.
                            structure = get_full_conventional_structure(get_cod_str(cif_content))
                            status_placeholder.markdown(
                                f"- **Structure loaded:** `{structure.composition.reduced_formula}` (cod_{entry.get('file')})")

                            combined_structures[f"{entry.get('file')}"] = structure
                        except Exception as e:
                            st.error(f"Error processing COD entry {entry.get('file')}: {e}")
                st.success(f"COD: Found {len(cod_entries)} structures.")
                length = len(cod_entries)
            else:
                st.warning("COD: No matching structures found.")
                length = 0
        else:
            st.error("Please enter at least one element for the COD search.")
        st.session_state.full_structures_see = combined_structures

if search_both:
    clear_candidates()
    # --- Materials Project Search ---
    with st.spinner(
            f"Searching **the MP database**, please wait. Note that especially the first search typically takes a bit longer while establishing initial connections. 😊"):
        try:
            with MPRester(MP_API_KEY) as mpr:
                elements_list = sorted(set(mp_search_query.split()))
                docs = mpr.materials.summary.search(
                    elements=elements_list,
                    num_elements=len(elements_list),
                    fields=["material_id", "formula_pretty", "symmetry"]
                )
                if docs:
                    status_placeholder = st.empty()
                    for doc in docs:
                        full_structure = mpr.get_structure_by_material_id(doc.material_id)
                        if convert_to_conventional:
                            structure_to_use = get_full_conventional_structure(full_structure, symprec=0.1)
                        elif pymatgen_prim_cell_lll:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                            structure_to_use = structure_to_use.get_reduced_structure(reduction_algo="LLL")
                        elif pymatgen_prim_cell_no_reduce:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                        else:
                            structure_to_use = full_structure
                        combined_structures[doc.material_id] = structure_to_use
                        status_placeholder.markdown(
                            f"- **Structure loaded:** `{structure_to_use.composition.reduced_formula}` ({doc.material_id})"
                        )
                    st.success(f"Materials Project: Found {len(docs)} structures.")
                    length_mp = len(docs)
                else:
                    st.warning("Materials Project: No matching structures found.")
                    length_mp = 0

        except Exception as e:
            st.error(f"Materials Project error: {e}")

    # --- AFLOW Search ---
    with st.spinner(
            f"Searching the **AFLOW database**, please wait.  Note that especially the first search typically takes a bit longer while establishing initial connections. 😊"):
        try:
            if aflow_elements_input:
                elements = re.split(r'[\s,]+', aflow_elements_input.strip())
                elements = [el for el in elements if el]
                ordered_elements = sorted(elements)
                ordered_str = ",".join(ordered_elements)
                aflow_nspecies = len(ordered_elements)
            else:
                ordered_str = ""
                aflow_nspecies = 0
            import aflow

            results = list(
                aflow.search(catalog="icsd")
                .filter((AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                .select(
                    AFLOW_K.auid,
                    AFLOW_K.compound,
                    AFLOW_K.geometry,
                    AFLOW_K.spacegroup_relax,
                    AFLOW_K.aurl,
                    AFLOW_K.files,
                )
            )
            if results:
                status_placeholder = st.empty()
                for entry in results:
                    structure = get_structure_from_aflow(entry)
                    structure = get_full_conventional_structure(structure)
                    if structure is not None:
                        combined_structures[entry.auid] = structure
                    status_placeholder.markdown(
                        f"- **Structure loaded:** `{structure.composition.reduced_formula}` (aflow_{entry.auid})"
                    )
                st.success(f"AFLOW: Found {len(results)} structures.")
                length_aflow = len(results)
            else:
                st.warning("AFLOW: No matching structures found.")
                length_aflow = 0
        except Exception as e:
            st.error(f"AFLOW error: {e}")
        st.session_state.full_structures_see = combined_structures

    with st.spinner(
            f"Searching **the COD database**, please wait. Note that especially the first search typically takes a bit longer while establishing initial connections. 😊"):
        # Build COD search parameters. Here we assume the user enters one or more elements.
        elements = [el.strip() for el in cod_elements_input.split() if el.strip()]
        if elements:
            params = {'format': 'json'}
            # Set search parameters for each element (COD expects keys like 'el1', 'el2', etc.)
            for i, el in enumerate(elements, start=1):
                params[f'el{i}'] = el
            # Set strict search to ensure the number of elements matches exactly.
            params['strictmin'] = str(len(elements))
            params['strictmax'] = str(len(elements))

            cod_entries = get_cod_entries(params)
            if cod_entries:
                status_placeholder = st.empty()
                for entry in cod_entries:
                    cif_content = get_cif_from_cod(entry)
                    if cif_content:
                        try:
                            # Use a unique key (e.g., prefix with "cod_") to avoid collision with MP/AFLOW IDs.
                            structure = get_full_conventional_structure(get_cod_str(cif_content))
                            status_placeholder.markdown(
                                f"- **Structure loaded:** `{structure.composition.reduced_formula}` (cod_{entry.get('file')})")

                            combined_structures[f"{entry.get('file')}"] = structure
                        except Exception as e:
                            st.error(f"Error processing COD entry {entry.get('file')}: {e}")
                st.success(f"COD: Found {len(cod_entries)} structures.")
                length_cod = len(cod_entries)
            else:
                st.warning("COD: No matching structures found.")
                length_cod = 0
            length = length_mp + length_aflow + length_cod
        else:
            st.error("Please enter at least one element for the COD search.")
        st.session_state.full_structures_see = combined_structures

elif search_mp:
    clear_candidates()
    with st.spinner(
            f"Searching the **MP database**, please wait. The first search typically takes a bit longer while establishing initial connections. 😊"):
        try:
            with MPRester(MP_API_KEY) as mpr:
                elements_list = sorted(set(mp_search_query.split()))
                docs = mpr.materials.summary.search(
                    elements=elements_list,
                    num_elements=len(elements_list),
                    fields=["material_id", "formula_pretty", "symmetry"]
                )
                if docs:
                    status_placeholder = st.empty()
                    for doc in docs:
                        full_structure = mpr.get_structure_by_material_id(doc.material_id)
                        if convert_to_conventional:
                            structure_to_use = get_full_conventional_structure(full_structure, symprec=0.1)
                        elif pymatgen_prim_cell_lll:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                            structure_to_use = structure_to_use.get_reduced_structure(reduction_algo="LLL")
                        elif pymatgen_prim_cell_no_reduce:
                            analyzer = SpacegroupAnalyzer(full_structure)
                            structure_to_use = analyzer.get_primitive_standard_structure()
                        else:
                            structure_to_use = full_structure
                        combined_structures[doc.material_id] = structure_to_use
                        status_placeholder.markdown(
                            f"- **Structure loaded:** `{structure_to_use.composition.reduced_formula}` ({doc.material_id})"
                        )
                    st.success(f"Materials Project: Found {len(docs)} structures.")
                    length = len(docs)
                else:
                    st.warning("Materials Project: No matching structures found.")
                    length = 0

        except Exception as e:
            st.error(f"Materials Project error: {e}")
        st.session_state.full_structures_see = combined_structures

elif search_aflow:
    clear_candidates()
    with st.spinner(
            f"Searching the **AFLOW database**, please wait. The first search typically takes a bit longer while establishing initial connections. 😊"):
        try:
            if aflow_elements_input:
                elements = re.split(r'[\s,]+', aflow_elements_input.strip())
                elements = [el for el in elements if el]
                ordered_elements = sorted(elements)
                ordered_str = ",".join(ordered_elements)
                aflow_nspecies = len(ordered_elements)
            else:
                ordered_str = ""
                aflow_nspecies = 0
            import aflow

            results = list(
                aflow.search(catalog="icsd")
                .filter((AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                .select(
                    AFLOW_K.auid,
                    AFLOW_K.compound,
                    AFLOW_K.geometry,
                    AFLOW_K.spacegroup_relax,
                    AFLOW_K.aurl,
                    AFLOW_K.files,
                )
            )
            if results:
                status_placeholder = st.empty()
                for entry in results:
                    structure = get_structure_from_aflow(entry)
                    structure = get_full_conventional_structure(structure)
                    if structure is not None:
                        combined_structures[entry.auid] = structure
                    status_placeholder.markdown(
                        f"- **Structure loaded:** `{structure.composition.reduced_formula}` (aflow_{entry.auid})"
                    )
                st.success(f"AFLOW: Found {len(results)} structures.")
                length = len(results)
            else:
                st.warning("AFLOW: No matching structures found.")
                length = 0
        except Exception as e:
            st.error(f"AFLOW error: {e}")
        st.session_state.full_structures_see = combined_structures

# --- XRD Pattern Calculation and Matching ---
st.subheader("Comparing Calculated XRD Patterns to Experimental Data")
compare_intensities = st.checkbox("Compare intensities as well", value=True)

min_intensity_threshold = st.slider("Minimum calculated peak intensity to consider", min_value=0, max_value=100,
                                    value=5)
tolerance_value = st.slider(f"# Tolerance for Peak Matching (d-spacing (Å))",
                            min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f")
st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
# Proceed only if there are structures retrieved and experimental data are valid.
if "full_structures_see" in st.session_state and st.session_state.full_structures_see:
    if experimental_peaks is not None and experimental_intensities is not None:

        # Build candidate list only if it is not already stored.
        if "candidate_list" not in st.session_state:
            status_placeholder = st.empty()
            status_messages = []
            candidates = []
            idxx = 1
            for material_id, structure in st.session_state.full_structures_see.items():
                try:
                    pattern = calculate_xrd_pattern(structure, wavelength=user_wavelength, range=full_range)
                    if compare_intensities:
                        score = get_peak_match_score_with_intensity(
                            experimental_peaks, experimental_intensities, pattern, user_wavelength,
                            w_angle=1, w_intensity=1, min_intensity=min_intensity_threshold
                        )
                        method = "combined d-spacing and intensity"
                    else:
                        score = get_peak_match_score(
                            experimental_peaks, pattern, user_wavelength, min_intensity=min_intensity_threshold
                        )
                        method = "d-spacing only"

                    comp = structure.composition.reduced_formula
                    lattice = structure.lattice
                    lattice_str = (f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å\n\n"
                                   f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")
                    space_group = SpacegroupAnalyzer(structure).get_space_group_symbol()
                    space_group_number = SpacegroupAnalyzer(structure).get_space_group_number()
                    with st.spinner("Comparing structures..."):
                        current_message = (
                            f"**Comparing structure: {idxx}/{length}** {material_id} ({comp}) using {method} method. "
                            f"Resulting match score: {score:.2f}")
                        status_messages.append(current_message)
                        status_placeholder.write(status_messages[-1])

                    candidates.append({
                        "id": material_id,
                        "score": score,
                        "structure": structure,
                        "pattern": pattern,
                        "composition": comp,
                        "lattice_str": lattice_str,
                        "space_group": space_group,
                        "space_group_number": space_group_number,
                        "lattice_volume": structure.lattice.volume,
                        "num_atoms": structure.num_sites
                    })
                    idxx = idxx + 1
                except Exception as e:
                    st.error(f"Could not calculate XRD pattern for {material_id}: {e}")

            if candidates:
                candidates_sorted = sorted(candidates, key=lambda x: x["score"])
                candidates_top = candidates_sorted[:30]
                for rank, cand in enumerate(candidates_top, start=1):
                    cand["rank"] = rank
                st.session_state.candidate_list = candidates_top
                st.session_state.candidate_index = 0  # initialize index
            else:
                st.warning("No candidate structures produced a calculable XRD pattern.")
        else:
            candidates_top = st.session_state.candidate_list

        if candidates_top:
            candidate_display_options = [
                (
                    f"{cand['rank']}. Score: {cand['score']:.2f} | ID: {cand['id']} | {cand['composition']} | {cand['space_group']} | {cand['lattice_str']}")
                for cand in candidates_top
            ]
            expander = st.expander("Show Top 30 Candidate Structures", expanded=True)
            selected_candidate_str = expander.selectbox(
                "Select a candidate",
                candidate_display_options,
                key="selected_candidate",  # store selection in session state
                on_change=update_candidate_index  # update candidate_index when changed
            )

            col_nav, col_plot = st.columns([1, 3])
            with col_nav:
                if st.button("Next Candidate", key="next_candidate"):
                    st.session_state.candidate_index = (st.session_state.candidate_index + 1) % len(candidates_top)
                if st.button("Previous Candidate", key="prev_candidate"):
                    st.session_state.candidate_index = (st.session_state.candidate_index - 1) % len(candidates_top)
                selected_candidate = candidates_top[st.session_state.candidate_index]
                st.markdown("### Selected Candidate")
                rank = selected_candidate["rank"]
                score = selected_candidate["score"]

                if rank == 1:
                    rank_style = "color:#DAA520; font-weight:600; font-size:19px;"  # soft gold
                    emoji = "🥇"
                elif rank == 2:
                    rank_style = "color:#C0C0C0; font-weight:600; font-size:18px;"  # soft silver
                    emoji = "🥈"
                elif rank == 3:
                    rank_style = "color:#b87333; font-weight:600; font-size:17px;"  # soft bronze
                    emoji = "🥉"
                else:
                    rank_style = "color:#333333; font-size:16px;"  # neutral dark gray
                    emoji = ""

                st.markdown(
                    f"<p style='{rank_style}'>{emoji} <b>Ranking:</b> {rank} "
                    f"(<b>Match Score:</b> {score:.2f})</p>",
                    unsafe_allow_html=True
                )
                st.write(f"**Composition:** {selected_candidate['composition']}")
                st.write(f"**ID:** {selected_candidate['id']}")
                if selected_candidate['id'].startswith("mp"):
                    linnk = f"https://materialsproject.org/materials/{selected_candidate['id']}"
                    down_name = 'mp'
                elif selected_candidate['id'].startswith("aflow"):
                    down_name = 'aflow'
                    linnk = f"https://aflowlib.duke.edu/search/ui/material/?id={selected_candidate['id']}"
                else:
                    linnk = f"https://www.crystallography.net/cod/{selected_candidate['id']}.html"
                    down_name = 'cod'


                st.write(f"**Lattice Parameters:** {selected_candidate['lattice_str']}")
                st.write("**Link:**", linnk)
                st.write(
                    f"**Space Group:** {selected_candidate['space_group']} ({selected_candidate['space_group_number']})")
                st.write(f"**Lattice Volume:** {selected_candidate['lattice_volume']:.2f} Å³")
                st.write(f"**Number of Atoms:** {selected_candidate['num_atoms']}")
                # --- Download CIF Button ---
                cif_writer = CifWriter(st.session_state.full_structures_see[selected_candidate['id']], symprec=0.01)

                cif_data = str(cif_writer)

                file_name = f"{selected_candidate['composition']}_{selected_candidate['space_group_number']}_{down_name}.cif"

                # Add a download button for the CIF file.
                st.download_button(
                    label="Download CIF",
                    data=cif_data,
                    file_name=file_name,
                    mime="chemical/x-cif",
                    type="primary",
                )
            selected_candidate = candidates_top[st.session_state.candidate_index]
            detail_intensity_filter = st.slider("Detailed Peak Intensity Filter: show only peaks with intensity ≥",
                                                min_value=0, max_value=100, value=0)

            # --- Interactive Plot (Placed at the top) ---
            st.markdown("### Interactive Visualization of Selected Candidate's XRD Pattern")
            pattern = selected_candidate["pattern"]
            vertical_x_calc = []
            vertical_y_calc = []
            vertical_hover_calc = []
            for i in range(len(pattern.x)):
                try:
                    hkls_for_peak = pattern.hkls[i]
                    d_spacing_val = pattern.d_hkls[i]
                    hkl_list = []
                    for refl in hkls_for_peak:
                        if "hkl" in refl:
                            hkl_str = "(" + " ".join(str(x) for x in refl["hkl"]) + ")"
                            hkl_list.append(hkl_str)
                    hkl_text = ", ".join(hkl_list)
                    d_spacing_str = f"{d_spacing_val:.4f} Å"
                    hover_text = f"<b>HKL: {hkl_text}</b><br>d-spacing = {d_spacing_str}"
                except Exception:
                    hkl_text = "N/A"
                two_theta_val = pattern.x[i]
                vertical_x_calc.extend([two_theta_val, two_theta_val, None])
                vertical_y_calc.extend([0, pattern.y[i], None])
                # vertical_hover_calc.extend([f"<b>hkl: {hkl_text}</b>", f"<b>hkl: {hkl_text}</b>", None])
                vertical_hover_calc.extend([hover_text, hover_text, None])
            trace_calc = go.Scatter(
                x=vertical_x_calc,
                y=vertical_y_calc,
                mode="lines",
                name="Calculated Peaks",
                line=dict(color="black", width=2, dash="solid"),
                hoverinfo="text",
                text=vertical_hover_calc,
                # hovertemplate="<b>Calculated Peak:</b><br>%{text}<br><b>2θ = %{x:.2f}°</b><br><b>Intensity = %{y:.2f}</b><extra></extra>"
                hovertemplate="<span style='color:black;'><b>Calculated Peak:</b><br>%{text}<br>2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>"
            )

            exp_assigned_x = []
            exp_assigned_y = []
            exp_assigned_hover = []
            exp_unassigned_x = []
            exp_unassigned_y = []
            exp_unassigned_hover = []

            valid_indices = [i for i, intensity in enumerate(pattern.y) if intensity >= detail_intensity_filter]
            assignment = {}
            assigned_exp_idxs = set()
            if valid_indices:
                calc_d_all = np.array(pattern.d_hkls)
                valid_calc_d = calc_d_all[valid_indices]
                exp_d = two_theta_to_d(experimental_peaks, user_wavelength)
                cost_matrix = np.abs(exp_d[:, None] - valid_calc_d[None, :])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for exp_idx, valid_idx in zip(row_ind, col_ind):
                    if cost_matrix[exp_idx, valid_idx] < tolerance_value:
                        calc_index = valid_indices[valid_idx]
                        assignment[calc_index] = {
                            "exp_angle": experimental_peaks[exp_idx],
                            "exp_intensity": experimental_intensities[exp_idx]
                        }
                        assigned_exp_idxs.add(exp_idx)
            for i, (exp_angle, exp_intensity) in enumerate(zip(experimental_peaks, experimental_intensities)):
                d_exp = two_theta_to_d(exp_angle, user_wavelength)
                hover_text = (f"<b>Exp Peak {i + 1}:</b><br>d-spacing = {d_exp:.4f} Å<br>"
                              f"2θ = {exp_angle:.2f}°<br>Intensity = {exp_intensity:.2f}")
                if i in assigned_exp_idxs:
                    exp_assigned_x.extend([exp_angle, exp_angle, None])
                    exp_assigned_y.extend([0, exp_intensity, None])
                    exp_assigned_hover.extend([hover_text, hover_text, None])
                else:
                    exp_unassigned_x.extend([exp_angle, exp_angle, None])
                    exp_unassigned_y.extend([0, exp_intensity, None])
                    exp_unassigned_hover.extend([hover_text, hover_text, None])
            trace_exp_assigned = go.Scatter(
                x=exp_assigned_x,
                y=exp_assigned_y,
                mode="lines+markers",
                name="Experimental Peaks (Matched)",
                line=dict(color="blue", width=2, dash="dot"),
                hoverinfo="text",
                text=exp_assigned_hover,
                # hovertemplate="<b>Matched Exp Peak:</b><br>%{text}<extra></extra>"
                hovertemplate="<span style='color:blue;'><b>Matched Exp Peak:</b><br>%{text}</span><extra></extra>"
            )
            trace_exp_unassigned = go.Scatter(
                x=exp_unassigned_x,
                y=exp_unassigned_y,
                mode="lines",
                name="Experimental Peaks (Unmatched)",
                line=dict(color="red", width=2, dash="dot"),
                hoverinfo="text",
                text=exp_unassigned_hover,
                # hovertemplate="<b>Unmatched Exp Peak:</b><br>%{text}<extra></extra>"
                hovertemplate="<span style='color:red;'><b>Unmatched Exp Peak:</b><br>%{text}</span><extra></extra>"
            )
            plot_data = [trace_calc, trace_exp_assigned, trace_exp_unassigned]

            # Add user's uploaded experimental data if available
            if uploaded_x is not None and uploaded_y is not None:
                user_trace = go.Scatter(
                    x=uploaded_x,
                    y=uploaded_y,
                    mode="lines+markers",
                    name="User Uploaded XRD",
                    line=dict(dash='solid', color="green", width=1),
                    # hovertemplate="<b>User XRD Data:</b><br>2θ = %{x:.2f}°<br>Intensity = %{y:.2f}<extra></extra>"
                    hovertemplate="<span style='color:green;'><b>User XRD Data:</b><br>2θ = %{x:.2f}°<br>Intensity = %{y:.2f}</span><extra></extra>",
                    marker=dict(color='green', size=3.5)
                )
                plot_data.append(user_trace)

            # Create the final figure
            fig = go.Figure(data=plot_data)
            fig.update_layout(
                height=900,
                title=dict(text="Interactive XRD Pattern (2θ)", font=dict(size=24)),
                xaxis=dict(title=dict(text="2θ (°)", font=dict(size=30, color="black")),
                           tickfont=dict(size=30, color="black")),
                yaxis=dict(title=dict(text="Intensity", font=dict(size=30, color="black")),
                           tickfont=dict(size=30, color="black"), showgrid=False, range=[-10, 120]),
                legend=dict(
                    font=dict(size=24, color="black"),
                    orientation="h",
                    y=-0.3,
                    x=0.5,
                    xanchor="center"
                ),
                template="plotly_white",
                hovermode='x',
                hoverlabel=dict(font=dict(size=20), )
            )

            # Display your interactive plot
            with col_plot:
                st.plotly_chart(fig, use_container_width=True)

            # --- Complete Calculated XRD Pattern Details (Below the Interactive Plot) ---
            st.markdown("#### Complete Calculated XRD Pattern (filtered by Detailed Peak Intensity)")
            pattern = selected_candidate["pattern"]
            valid_indices = [i for i, intensity in enumerate(pattern.y) if intensity >= detail_intensity_filter]
            assignment = {}
            assigned_exp_idxs = set()
            if valid_indices:
                calc_d_all = np.array(pattern.d_hkls)
                valid_calc_d = calc_d_all[valid_indices]
                exp_d = two_theta_to_d(experimental_peaks, user_wavelength)
                cost_matrix = np.abs(exp_d[:, None] - valid_calc_d[None, :])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for exp_idx, valid_idx in zip(row_ind, col_ind):
                    if cost_matrix[exp_idx, valid_idx] < tolerance_value:
                        calc_index = valid_indices[valid_idx]
                        assignment[calc_index] = {
                            "exp_angle": experimental_peaks[exp_idx],
                            "exp_intensity": experimental_intensities[exp_idx]
                        }
                        assigned_exp_idxs.add(exp_idx)
            displayed_peak = False
            for i, (calc_angle, calc_intensity) in enumerate(zip(pattern.x, pattern.y), start=1):
                if calc_intensity < detail_intensity_filter:
                    continue
                calc_d = np.array(pattern.d_hkls)[i - 1]
                if (i - 1) in assignment:
                    exp_info = assignment[i - 1]
                    exp_d_assigned = two_theta_to_d(exp_info['exp_angle'], user_wavelength)
                    st.markdown(
                        f"<div>"
                        f"<b>Peak {i}:</b> 2θ = {calc_angle:.2f}°  |  d-spacing = {calc_d:.4f} Å "
                        f"(<b style='color:blue;'>Exp 2θ = {exp_info['exp_angle']:.2f}° | Exp d-spacing = {exp_d_assigned:.4f} Å</b>), "
                        f"<b>I = {calc_intensity:.2f}</b> "
                        f"(<b style='color:blue;'>Exp I = {exp_info['exp_intensity']:.2f}</b>)"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div><b>Peak {i}:</b> 2θ = {calc_angle:.2f}°  |  d-spacing = {calc_d:.4f} Å, "
                        f"<b>I = {calc_intensity:.2f}</b></div>",
                        unsafe_allow_html=True
                    )
                displayed_peak = True
            if not displayed_peak:
                st.warning("No peaks above the selected detailed intensity threshold were found.")

    else:
        st.error("Please check your experimental data inputs.")
else:
    st.info("Please run a database search to retrieve structures.")
