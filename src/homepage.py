import streamlit as st
from pathlib import Path

st.set_page_config(page_title="homepage", page_icon="🏠")

st.title("Office for National Statistics (ONS) Data Science Campus (DSC) App")

repo_dir = Path.cwd().parent
img_dir = repo_dir.joinpath("images")
st.logo(str(img_dir.joinpath("final_logo.png")))


def app():
    with st.expander("Summary:"):
        st.write(
            """The Data Science Campus at the Office for National Statistics (ONS)
                    worked to automate the detection of shelter footprints in Somali
                    internally displaced people (IDP) camps. We used very high-resolution (VHR)
                    to train our model which capable of detecting tents, informal, and formal
                    buildings to a high degree of accuracy. The work of the DSC means IDP camp footprints
                    can be created in minutes compared to months."""
        )

    st.info(
        """Please be aware that The Data Science Campus at
                the Office for National Statistics (ONS) **will not**
                update or maintain this application.

                For more details on this project please use the link below"""
    )

    st.link_button(
        "Information",
        help="Click here",
        url="https://github.com/datasciencecampus/somalia_unfpa_census_support/blob/main/README.md",
    )

if __name__ == "__main__":
    app()
