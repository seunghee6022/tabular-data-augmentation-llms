"""
Data loading and preprocessing for California housing, insurance, and adult datasets
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional, List, Any, Union
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.great_format_utils import GReaTFormatHandler, create_dynamic_prompt

class DatasetLoader:
    """Load and preprocess datasets for LLM fine-tuning"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # TextDatasetLoader is missing, so we cannot initialize it here.
        self.tabular_datasets = ['california', 'adult', 'insurance']
    
    def create_great_format(self, row: pd.Series, column_mapping: Dict[str, str] = None, 
                           precision_mapping: Dict[str, int] = None) -> str:
        """
        Dynamically create GReaT format string from column names and values
        
        Args:
            row: pandas Series with data
            column_mapping: Optional mapping of original column names to display names
            precision_mapping: Optional mapping of column names to decimal precision
        
        Returns:
            GReaT formatted string: "col1 is val1, col2 is val2, ..."
        """
        great_parts = []
        
        for col_name in row.index:
            # Get display name (use mapping if provided, otherwise use original)
            display_name = column_mapping.get(col_name, col_name) if column_mapping else col_name
            
            # Get value
            value = row[col_name]
            
            # Format value based on type and precision mapping
            if pd.isna(value):
                formatted_value = "unknown"
            elif isinstance(value, (int, np.integer)):
                formatted_value = str(int(value))
            elif isinstance(value, (float, np.floating)):
                # Use precision mapping if provided
                precision = precision_mapping.get(col_name, 2) if precision_mapping else 2
                formatted_value = f"{float(value):.{precision}f}"
            else:
                # String or categorical
                formatted_value = str(value)
            
            great_parts.append(f"{display_name} is {formatted_value}")
        
        return ", ".join(great_parts)
    
    def create_dynamic_prompt(self, row: pd.Series, dataset_name: str, 
                             column_mapping: Dict[str, str] = None,
                             precision_mapping: Dict[str, int] = None) -> Dict[str, str]:
        """
        Create dynamic prompt using GReaT format
        
        Args:
            row: pandas Series with data
            dataset_name: Name of the dataset for prompt context
            column_mapping: Optional column name mapping
            precision_mapping: Optional precision mapping
        
        Returns:
            Dictionary with instruction, response, and input
        """
        great_format = self.create_great_format(row, column_mapping, precision_mapping)
        
        prompt = f"""Generate a new synthetic {dataset_name} record. The generated record must be in exactly the same format as this example:
{row.to_dict()}

Generate a similar but different {dataset_name} record:"""
        
        return {
            "instruction": prompt,
            "response": great_format,
            "input": ""
        }
        
    def load_california_housing(self) -> DatasetDict:
        """Load California housing dataset"""
        print("Loading California housing dataset...")
        
        # Load the dataset
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        
        # Create dataset configuration using utility function with rich metadata
        housing_config = GReaTFormatHandler.create_dataset_config(
            df,
            dataset_name="California housing",
            target_column="target",
            column_mapping={'target': 'MedHouseVal'},
            precision_mapping={
                'MedInc': 2, 'HouseAge': 1, 'AveRooms': 1, 'AveBedrms': 1,
                'Population': 0, 'AveOccup': 1, 'Latitude': 2, 'Longitude': 2, 'target': 2
            },
            dataset_description="California housing dataset containing information about housing districts in California from the 1990 census. Used to predict median house values based on demographic and geographic features.",
            column_descriptions={
                'MedInc': 'Median income in block group (in tens of thousands of dollars)',
                'HouseAge': 'Median house age in block group (in years)',
                'AveRooms': 'Average number of rooms per household',
                'AveBedrms': 'Average number of bedrooms per household',
                'Population': 'Total population in block group',
                'AveOccup': 'Average number of household members',
                'Latitude': 'Geographic latitude of the block group',
                'Longitude': 'Geographic longitude of the block group',
                'MedHouseVal': 'Median house value in block group (in hundreds of thousands of dollars)'
            },
            domain_context="Real estate and demographic data from California census blocks, useful for housing market analysis and property valuation modeling."
        )
        
        # Create text descriptions for fine-tuning using dynamic format
        def create_housing_prompt(row):
            return create_dynamic_prompt(row, housing_config)
        
        # Apply transformation
        housing_data = df.apply(create_housing_prompt, axis=1).tolist()
        
        # Split data
        train_data, test_data = train_test_split(housing_data, test_size=0.2, random_state=42)
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
    
    def load_adult_dataset(self) -> DatasetDict:
        """Load Adult (Census Income) dataset"""
        print("Loading Adult dataset...")
        
        # Download if not exists
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # Load training data
        try:
            train_df = pd.read_csv(train_url, names=columns, skipinitialspace=True)
        except:
            print("Could not download from UCI. Using sample data...")
            # Create sample data if download fails
            train_df = self._create_sample_adult_data()
        
        # Clean data
        train_df = train_df.replace('?', np.nan).dropna()
        train_df['income'] = train_df['income'].str.strip().str.rstrip('.')
        
        # Create dataset configuration using utility function with rich metadata
        adult_config = GReaTFormatHandler.create_dataset_config(
            train_df,
            dataset_name="adult census",
            target_column="income",
            column_mapping={
                'education-num': 'Education-num', 'marital-status': 'Marital-status',
                'capital-gain': 'Capital-gain', 'capital-loss': 'Capital-loss',
                'hours-per-week': 'Hours-per-week', 'native-country': 'Native-country'
            },
            dataset_description="Adult Census Income dataset from the 1994 Census database. Contains demographic and employment information to predict whether a person's income exceeds $50K/year.",
            column_descriptions={
                'age': 'Age of the individual in years',
                'workclass': 'Type of employment (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)',
                'fnlwgt': 'Final weight assigned by Census Bureau (number of people the census believes the entry represents)',
                'education': 'Highest level of education completed',
                'Education-num': 'Numerical encoding of education level (higher numbers = more education)',
                'Marital-status': 'Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)',
                'occupation': 'Type of occupation/job',
                'relationship': 'Relationship status within household (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)',
                'race': 'Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)',
                'sex': 'Gender (Female, Male)',
                'Capital-gain': 'Capital gains income (investment income)',
                'Capital-loss': 'Capital losses (investment losses)',
                'Hours-per-week': 'Number of hours worked per week',
                'Native-country': 'Country of origin',
                'income': 'Income level (<=50K or >50K per year)'
            },
            domain_context="Socioeconomic and demographic data for income prediction, commonly used in machine learning for classification tasks and fairness analysis."
        )
        
        def create_adult_prompt(row):
            return create_dynamic_prompt(row, adult_config)
        
        # Apply transformation
        adult_data = train_df.apply(create_adult_prompt, axis=1).tolist()
        
        # Split data
        train_data, test_data = train_test_split(adult_data, test_size=0.2, random_state=42)
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
    
    def load_insurance_dataset(self) -> DatasetDict:
        """Load insurance dataset (using sample data)"""
        print("Loading Insurance dataset...")
        
        # Create sample insurance data (you can replace with actual dataset URL)
        insurance_data = self._create_sample_insurance_data()
        
        # Create dataset configuration using utility function with rich metadata
        insurance_config = GReaTFormatHandler.create_dataset_config(
            insurance_data,
            dataset_name="insurance",
            target_column="charges",
            precision_mapping={
                'age': 0, 'bmi': 1, 'children': 0, 'charges': 2
            },
            dataset_description="Health insurance dataset containing personal information and medical insurance charges. Used to predict insurance costs based on individual characteristics.",
            column_descriptions={
                'age': 'Age of the insurance beneficiary in years',
                'sex': 'Gender of the insurance beneficiary (male/female)',
                'bmi': 'Body Mass Index (BMI) - measure of body fat based on height and weight (kg/m²)',
                'children': 'Number of children/dependents covered by health insurance',
                'smoker': 'Smoking status (yes/no) - whether the beneficiary smokes tobacco',
                'region': 'Geographic region in the US (southwest, southeast, northwest, northeast)',
                'charges': 'Individual medical costs billed by health insurance (in US dollars)'
            },
            domain_context="Healthcare and insurance industry data for risk assessment and premium calculation. Smoking status and BMI are major factors in determining insurance costs."
        )
        
        def create_insurance_prompt(row):
            return create_dynamic_prompt(row, insurance_config)
        
        # Apply transformation
        insurance_prompts = insurance_data.apply(create_insurance_prompt, axis=1).tolist()
        
        # Split data
        train_data, test_data = train_test_split(insurance_prompts, test_size=0.2, random_state=42)
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
    
    def _create_sample_adult_data(self) -> pd.DataFrame:
        """Create sample adult dataset if download fails"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(17, 90, n_samples),
            'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov'], n_samples),
            'fnlwgt': np.random.randint(12285, 1484705, n_samples),
            'education': np.random.choice(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm'], n_samples),
            'education-num': np.random.randint(1, 16, n_samples),
            'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated'], n_samples),
            'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial'], n_samples),
            'relationship': np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative'], n_samples),
            'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], n_samples),
            'sex': np.random.choice(['Female', 'Male'], n_samples),
            'capital-gain': np.random.randint(0, 99999, n_samples),
            'capital-loss': np.random.randint(0, 4356, n_samples),
            'hours-per-week': np.random.randint(1, 99, n_samples),
            'native-country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada'], n_samples),
            'income': np.random.choice(['<=50K', '>50K'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_insurance_data(self) -> pd.DataFrame:
        """Create sample insurance dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        ages = np.random.randint(18, 65, n_samples)
        bmis = np.random.normal(30, 6, n_samples)
        bmis = np.clip(bmis, 15, 50)
        
        data = {
            'age': ages,
            'sex': np.random.choice(['male', 'female'], n_samples),
            'bmi': bmis,
            'children': np.random.randint(0, 6, n_samples),
            'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples)
        }
        
        # Create realistic charges based on features
        charges = []
        for i in range(n_samples):
            base_charge = 1000 + (data['age'][i] * 50)
            if data['smoker'][i] == 'yes':
                base_charge *= 2.5
            if data['bmi'][i] > 35:
                base_charge *= 1.3
            base_charge += data['children'][i] * 500
            base_charge += np.random.normal(0, 1000)
            charges.append(max(base_charge, 1000))
        
        data['charges'] = charges
        
        return pd.DataFrame(data)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available individual datasets"""
        return self.tabular_datasets
    
    def load_dataset_by_name(self, dataset_name: str) -> DatasetDict:
        """Load a specific dataset by name"""
        if dataset_name in self.tabular_datasets:
            # Load tabular datasets
            if dataset_name == 'california':
                return self.load_california_housing()
            elif dataset_name == 'adult':
                return self.load_adult_dataset()
            elif dataset_name == 'insurance':
                return self.load_insurance_dataset()
        # Text datasets not implemented yet
        # elif dataset_name in self.text_datasets:
        #     return self.text_loader.load_dataset_by_name(dataset_name)
        else:
            available = self.get_available_datasets()
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

def format_instruction_dataset(example):
    """Format dataset for instruction tuning"""
    if example.get('input', '').strip():
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    
    example['text'] = prompt + example['response']
    return example
