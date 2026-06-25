import streamlit as st # type: ignore
import pandas as pd # type: ignore

from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# =====================
# LOAD DATASET
# =====================

@st.cache_data
def load_data():

    makanan = pd.read_excel(
        "data_makanan.xlsx",
        sheet_name="Menu"
    )

    makanan = makanan.loc[
        :,
        ~makanan.columns.str.contains("^Unnamed")
    ]

    makanan = makanan.dropna(
        how="all"
    ).reset_index(drop=True)

    makanan.columns = [

        "nama_menu",

        "kalori",

        "protein",

        "lemak",

        "karbohidrat"

    ]

    return makanan


makanan = load_data()

# =====================
# FUNGSI REKOMENDASI
# =====================

def rekomendasi(
    makanan,
    usia,
    jenis_kelamin,
    berat_badan,
    tinggi_badan,
    aktivitas
):

    faktor = {

        "Ringan": 1.375,

        "Sedang": 1.55,

        "Berat": 1.725

    }

    # BMR Mifflin-St Jeor

    if jenis_kelamin == "L":

        bmr = (

            (10 * berat_badan)

            + (6.25 * tinggi_badan)

            - (5 * usia)

            + 5

        )

    else:

        bmr = (

            (10 * berat_badan)

            + (6.25 * tinggi_badan)

            - (5 * usia)

            - 161

        )

    # TDEE

    tdee = bmr * faktor[aktivitas]

    # Target Kalori per Makan

    target_kalori = tdee / 3

    # Target Makronutrien

    target_protein = (
        target_kalori * 0.15
    ) / 4

    target_lemak = (
        target_kalori * 0.25
    ) / 9

    target_karbohidrat = (
        target_kalori * 0.60
    ) / 4

    # Filter Kandidat Berdasarkan Kalori

    toleransi = target_kalori * 0.30

    kandidat = makanan[
        (
            makanan["kalori"]
            - target_kalori
        ).abs()
        <= toleransi
    ].copy()

    if len(kandidat) < 5:

        kandidat = makanan.copy()

    # Profil User

    profil_user = [[

        target_protein,

        target_lemak,

        target_karbohidrat

    ]]

    fitur = [

        "protein",

        "lemak",

        "karbohidrat"

    ]

    scaler = MinMaxScaler()

    kandidat_scaled = scaler.fit_transform(
        kandidat[fitur]
    )

    profil_scaled = scaler.transform(
        profil_user
    )

    similarity = cosine_similarity(

        profil_scaled,

        kandidat_scaled

    )

    kandidat["Similarity"] = (
        similarity.flatten()
    )

    top5 = kandidat.sort_values(

        by="Similarity",

        ascending=False

    ).head(5)

    return (
        top5,
        tdee,
        target_kalori
    )


# =====================
# STREAMLIT UI
# =====================

st.title(
    "🍽️ Sistem Rekomendasi Menu Makanan Harian"
)

st.markdown(
    """
    Sistem rekomendasi menu makanan harian
    berdasarkan kebutuhan kalori menggunakan
    Content-Based Filtering dan Cosine Similarity.
    """
)

usia = st.number_input(
    "Usia",
    min_value=15,
    max_value=80,
    value=23
)

jenis_kelamin = st.selectbox(
    "Jenis Kelamin",
    ["L", "P"]
)

berat_badan = st.number_input(
    "Berat Badan (kg)",
    min_value=30,
    max_value=200,
    value=55
)

tinggi_badan = st.number_input(
    "Tinggi Badan (cm)",
    min_value=100,
    max_value=250,
    value=170
)

aktivitas = st.selectbox(
    "Aktivitas",
    [
        "Ringan",
        "Sedang",
        "Berat"
    ]
)

if st.button(
    "Tampilkan Rekomendasi"
):

    top5, tdee, target_kalori = rekomendasi(

        makanan,

        usia,

        jenis_kelamin,

        berat_badan,

        tinggi_badan,

        aktivitas

    )

    st.success(
        "Rekomendasi berhasil dibuat"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "TDEE",
            f"{tdee:.2f} kkal"
        )

    with col2:

        st.metric(
            "Target Kalori per Makan",
            f"{target_kalori:.2f} kkal"
        )

    st.subheader(
        "Top 5 Rekomendasi Menu"
    )

    st.dataframe(
        top5[
            [
                "nama_menu",
                "kalori",
                "protein",
                "lemak",
                "karbohidrat",
                "Similarity"
            ]
        ],
        use_container_width=True
    )