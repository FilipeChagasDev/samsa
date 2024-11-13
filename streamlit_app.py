import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd


def init_session_states():
    if 'selected_locations' not in st.session_state:
        st.session_state["selected_locations"] = []


def mk_intro_section():
    st.title('𓆣 Samsa')
    st.markdown('**Developed by Filipe Chagas Ferraz [(visit my website)](https://filipechagasdev.github.io/FilipeChagasDev/)**')
    st.caption('''
        Samsa is an Streamlit app that demonstrates the work of metaheuristic methods for solving the Traveling Salesman Problem.
               
        The Traveling Salesman Problem (TSP) is a classic computational problem that involves finding the shortest or fastest route 
        that passes through a set of points {p₁, p₂, ..., pₙ} on a map. This is an NP-hard problem, meaning there are no known 
        algorithms that solve it exactly in sub-exponential time. In practice, we use metaheuristic algorithms to solve the TSP, 
        which don’t necessarily provide exact solutions but do find good solutions efficiently. Some examples of metaheuristics that 
        can be used to solve TSP are: Ant Colony Optimization, Simulated Annealing, Hill Climbing, etc.
                    
        In this app, you can create an instance of the Traveling Salesman Problem, choose and parameterize a metaheuristic, and get 
        a complete visualization of the solution found and the search process performed by the metaheuristic. Have fun!
    ''')


def load_points_from_csv(f):
    points_df = pd.read_csv(f)
    st.session_state['selected_locations'] = []
    for i, row in points_df.iterrows():
        st.session_state['selected_locations'].append({
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Name': row['Name']
        })


def points_to_csv():
   df = pd.DataFrame(st.session_state['selected_locations'])
   return df.to_csv(index=False).encode('utf-8')


def mk_stopping_points_selection_section():
    col_map, col_data = st.columns(2)
    with col_map:
        st.write('Click on a location on the map to select it')
        map_display = folium.Map(location=[0, 0], zoom_start=1)

        for loc in st.session_state['selected_locations']:
            folium.Marker([loc['Latitude'], loc['Longitude']], popup='Selected point').add_to(map_display)

        map_display.add_child(folium.LatLngPopup())
        output = st_folium(map_display, width=700, height=500)

    with col_data:
        st.write('Selected points')

        col_save_btn, col_clear_btn = st.columns(2)     
        
        if len(st.session_state['selected_locations']) > 0:
            # Save CSV Button
            col_save_btn.download_button(
                label='Save CSV',
                data=points_to_csv(),
                file_name='samsa_points.csv',
                mime = 'text/csv',
                use_container_width=True
            )

            # Clear Points Button
            if col_clear_btn.button('Clear points', use_container_width=True):
                st.session_state['selected_locations'] = []
                st.rerun()
        
        if output['last_clicked'] is not None:
            lat, lon = output['last_clicked']['lat'], output['last_clicked']['lng']
            col_name_txt, col_add_btn = st.columns([2,1], vertical_alignment='bottom')

            # Point Name Input
            point_name = col_name_txt.text_input('Point name')
            
            # Add Point Button
            if col_add_btn.button('Add selected point', use_container_width=True):
                st.session_state['selected_locations'].append({'Latitude': lat, 'Longitude': lon, 'Name': point_name})
                st.rerun()
        
        # Points table
        st.table(st.session_state['selected_locations'])

        # "At least 4 points" warning
        if len(st.session_state['selected_locations']) < 4:
            st.warning('You should select at least 4 points!')


def mk_stopping_points_upload_section():
    points_csv_file = st.file_uploader('Upload the CSV file', type={'csv'})
    if points_csv_file is not None:
        load_points_from_csv(points_csv_file)
        mk_stopping_points_selection_section()


def mk_stopping_points_section():
    with st.container(border=True):
        st.subheader('Traveling salesman stopping points')
        st.caption('''
            In this step, you must select the locations where the traveling salesman should pass. 
            There are two ways to do this:
            * In "manually select points" mode, you must click on a point on the map, give it a name and click the add button.
            * In the "upload a CSV file with points" mode, you must upload a CSV file with the latitudes, logitudes and names 
              of the selected points, and you can also add points manually later.
        ''')
        
        option = st.selectbox('Mode', options=['Manually select points', 'Upload a CSV file with the points'])
        if option == 'Manually select points':
            mk_stopping_points_selection_section()
        else:
            mk_stopping_points_upload_section()


def mk_ant_colony_parameters_section():
    with st.container(border=True):
        st.caption('''
            The Ant Colony Optimization (ACO) algorithm for the Traveling Salesman Problem (TSP) is a metaheuristic 
            inspired by the natural behavior of ants searching for food. Each ant builds a route by moving from one 
            point to another, influenced by pheromone trails and the distance between points. Just as real ants tend 
            to follow paths with stronger pheromone concentrations, the artificial ants in ACO are more likely to 
            choose routes that others have taken, while also favoring shorter distances to balance exploration and 
            exploitation. After each iteration, pheromone trails evaporate at a set rate, preventing the algorithm 
            from getting stuck on suboptimal solutions. Over time, the best route becomes more reinforced, guiding 
            the colony toward an efficient solution to the TSP.
        ''')

        st.write('Parameters:')

        col_n_epochs_input, col_n_epochs_text = st.columns(2, vertical_alignment='bottom')
        n_epochs = col_n_epochs_input.number_input('Number of optimization epochs', min_value=10, step=1)
        col_n_epochs_text.caption('The number of epochs is the number of times the ant colony will explore the search space. The more points there are, the higher this number should be.')

        col_n_ants_input, col_n_ants_text = st.columns(2, vertical_alignment='bottom')
        n_ants = col_n_ants_input.number_input('Number of ants', min_value=2, step=1)
        col_n_ants_text.caption('This determines the number of ants exploring the search space in each epoch. The higher this number, the fewer times it will take and the better the results will be.')

        col_alpha_input, col_alpha_text = st.columns(2, vertical_alignment='bottom')
        alpha = col_alpha_input.number_input('Alpha ($\\alpha$)', value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        col_alpha_text.caption('$\\alpha$ (Pheromone Influence): This parameter controls the weight of pheromone in the ants decision-making. A higher value of $\\alpha$ makes ants more likely to follow trails with a higher concentration of pheromone, favoring previously explored routes.')

        col_beta_input, col_beta_text = st.columns(2, vertical_alignment='bottom')
        beta = col_beta_input.number_input('Beta ($\\beta$)', value=2.0, min_value=0.1, max_value=10.0, step=0.1)
        col_beta_text.caption('$\\beta$ (Visibility Influence): This parameter determines the importance of visibility, which is the inverse of the distance between points, in the decision process. A higher $\\beta$ value means that ants will prefer shorter paths, regardless of the amount of pheromone.')
        
        col_rho_input, col_rho_text = st.columns(2, vertical_alignment='bottom')
        rho = col_rho_input.number_input('Rho ($\\rho$)', value=0.1, min_value=0.01, max_value=0.99, step=0.01)
        col_rho_text.caption('$\\rho$ (Pheromone Evaporation Rate): This parameter controls the rate at which pheromone evaporates over time. A higher ρ value means that pheromone trails fade more quickly, which helps prevent the algorithm from getting trapped in suboptimal paths by allowing for more exploration of new routes. A lower $\\rho$ value means that pheromone persists longer, reinforcing paths that have been followed and encouraging ants to exploit these paths.')

        col_zq_input, col_zq_text = st.columns(2, vertical_alignment='bottom')
        zq = col_zq_input.number_input('$Z_Q$', value=2.0, min_value=-5.0, max_value=5.0, step=0.5)
        col_zq_text.caption('In this implementation, the pheromone trails are incremented as $\\tau_{i,j} = \\tau_{i,j} + \\frac{\\mu N - Z_Q \\sigma \\sqrt{N}}{L}$, where $\\mu$ is the average of all distances between pairs of points, $\\sigma$ is the standard deviation of these distances, $N$ is the number of points, and $L$ is the length of the best route found. The higher the $Z_Q$ value, the lower the pheromone increase each epoch.')

        st.session_state['metaheuristic'] = {
            'option': 'ACO',
            'params': {
                'n_epochs': n_epochs,
                'n_ants': n_ants,
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'zq': zq
            }
        }


def mk_metaheuristic_selection_section():
    with st.container(border=True):
        st.subheader('Metaheuristic selection')
        st.caption('In this step, you must choose a metaheuristic algorithm to solve the TSP instance and define the parameters of that algorithm.')
        
        option = st.selectbox('Metaheuristic option', options=[
            'Ant Colony Optimization (ACO)'
        ])

        if option == 'Ant Colony Optimization (ACO)':
            mk_ant_colony_parameters_section()

        if st.button('Run optimization'):
            if len(st.session_state['selected_locations']) < 4:
                st.error('The TSP instance should have at least 4 points')         


def main():
    st.set_page_config(
        page_title="Samsa",
        page_icon="𓆣",
        layout="wide"
    )
    init_session_states()
    mk_intro_section()
    mk_stopping_points_section()
    mk_metaheuristic_selection_section()


main()