"""
Utility functions for dynamic GReaT format handling
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union

class GReaTFormatHandler:
    """Handle dynamic GReaT format creation and parsing"""
    
    @staticmethod
    def create_great_format(row: Union[pd.Series, Dict], 
                           column_mapping: Dict[str, str] = None, 
                           precision_mapping: Dict[str, int] = None,
                           column_order: List[str] = None) -> str:
        """
        Dynamically create GReaT format string from column names and values
        
        Args:
            row: pandas Series or dictionary with data
            column_mapping: Optional mapping of original column names to display names
            precision_mapping: Optional mapping of column names to decimal precision
            column_order: Optional list to specify column order
        
        Returns:
            GReaT formatted string: "col1 is val1, col2 is val2, ..."
        """
        great_parts = []
        
        # Determine columns to iterate over
        if column_order:
            columns = column_order
        elif isinstance(row, pd.Series):
            columns = row.index.tolist()
        else:
            columns = list(row.keys())
        
        for col_name in columns:
            # Skip if column doesn't exist in row
            if col_name not in row:
                continue
                
            # Get display name (use mapping if provided, otherwise use original)
            display_name = column_mapping.get(col_name, col_name) if column_mapping else col_name
            
            # Get value
            value = row[col_name]
            
            # Format value based on type and precision mapping
            formatted_value = GReaTFormatHandler._format_value(value, col_name, precision_mapping)
            
            great_parts.append(f"{display_name} is {formatted_value}")
        
        return ", ".join(great_parts)
    
    @staticmethod
    def _format_value(value: Any, col_name: str, precision_mapping: Dict[str, int] = None) -> str:
        """Format a single value for GReaT format"""
        if pd.isna(value):
            return "unknown"
        elif isinstance(value, (int, np.integer)):
            return str(int(value))
        elif isinstance(value, (float, np.floating)):
            # Use precision mapping if provided
            precision = precision_mapping.get(col_name, 2) if precision_mapping else 2
            return f"{float(value):.{precision}f}"
        else:
            # String or categorical
            return str(value).strip()
    
    @staticmethod
    def create_parsing_patterns(columns: List[str], 
                               column_mapping: Dict[str, str] = None) -> Dict[str, str]:
        """
        Dynamically create regex patterns for parsing GReaT format
        
        Args:
            columns: List of column names
            column_mapping: Optional mapping of original to display names
        
        Returns:
            Dictionary mapping column names to regex patterns
        """
        patterns = {}
        
        for col_name in columns:
            # Get display name
            display_name = column_mapping.get(col_name, col_name) if column_mapping else col_name
            
            # Escape special regex characters in display name
            escaped_name = re.escape(display_name)
            
            # Create pattern for "DisplayName is value"
            patterns[col_name] = rf'{escaped_name} is\s*([^,\n]+?)(?=,|$)'
        
        return patterns
    
    @staticmethod
    def parse_great_format(text: str, 
                          columns: List[str],
                          column_mapping: Dict[str, str] = None,
                          type_mapping: Dict[str, type] = None) -> Optional[Dict[str, Any]]:
        """
        Parse GReaT format text back to dictionary
        
        Args:
            text: GReaT formatted text
            columns: Expected column names
            column_mapping: Optional column name mapping
            type_mapping: Optional type conversion mapping
        
        Returns:
            Dictionary with parsed values or None if parsing fails
        """
        patterns = GReaTFormatHandler.create_parsing_patterns(columns, column_mapping)
        record = {}
        
        for col_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                
                # Convert to appropriate type
                try:
                    if type_mapping and col_name in type_mapping:
                        target_type = type_mapping[col_name]
                        if target_type == int:
                            record[col_name] = int(float(value_str))  # Handle "1.0" -> 1
                        elif target_type == float:
                            record[col_name] = float(value_str)
                        else:
                            record[col_name] = target_type(value_str)
                    else:
                        # Try to infer type
                        if value_str.lower() in ['unknown', 'nan', 'null']:
                            record[col_name] = None
                        elif value_str.replace('.', '').replace('-', '').isdigit():
                            if '.' in value_str:
                                record[col_name] = float(value_str)
                            else:
                                record[col_name] = int(value_str)
                        else:
                            record[col_name] = value_str
                except (ValueError, TypeError):
                    record[col_name] = value_str  # Keep as string if conversion fails
            else:
                return None  # Missing required field
        
        return record if len(record) == len(columns) else None
    
    @staticmethod
    def create_dataset_config(df: pd.DataFrame, 
                             dataset_name: str,
                             target_column: str = None,
                             column_mapping: Dict[str, str] = None,
                             precision_mapping: Dict[str, int] = None,
                             dataset_description: str = None,
                             column_descriptions: Dict[str, str] = None,
                             domain_context: str = None) -> Dict[str, Any]:
        """
        Create a complete configuration for a dataset with rich metadata
        
        Args:
            df: DataFrame with the dataset
            dataset_name: Name of the dataset
            target_column: Name of target column (if any)
            column_mapping: Optional column name mapping
            precision_mapping: Optional precision mapping
            dataset_description: Detailed description of the dataset
            column_descriptions: Descriptions for each column
            domain_context: Additional domain context
        
        Returns:
            Configuration dictionary with metadata
        """
        columns = df.columns.tolist()
        
        # Auto-generate precision mapping if not provided
        if precision_mapping is None:
            precision_mapping = {}
            for col in columns:
                if df[col].dtype in ['int64', 'int32']:
                    precision_mapping[col] = 0
                elif df[col].dtype in ['float64', 'float32']:
                    # Determine appropriate precision based on data
                    if col.lower() in ['price', 'cost', 'charge', 'income', 'salary']:
                        precision_mapping[col] = 2  # Currency
                    elif col.lower() in ['latitude', 'longitude', 'coord']:
                        precision_mapping[col] = 2  # Coordinates
                    else:
                        precision_mapping[col] = 1  # General float
        
        # Auto-generate type mapping for parsing
        type_mapping = {}
        for col in columns:
            if df[col].dtype in ['int64', 'int32']:
                type_mapping[col] = int
            elif df[col].dtype in ['float64', 'float32']:
                type_mapping[col] = float
            else:
                type_mapping[col] = str
        
        return {
            'dataset_name': dataset_name,
            'columns': columns,
            'column_mapping': column_mapping or {},
            'precision_mapping': precision_mapping,
            'type_mapping': type_mapping,
            'target_column': target_column,
            'parsing_patterns': GReaTFormatHandler.create_parsing_patterns(columns, column_mapping),
            'dataset_description': dataset_description or f"Dataset containing {dataset_name} records",
            'column_descriptions': column_descriptions or {},
            'domain_context': domain_context or f"Data from the {dataset_name} domain"
        }

def create_dynamic_prompt(row: Union[pd.Series, Dict], config: Dict[str, Any]) -> Dict[str, str]:
    """
    Create dynamic prompt with rich metadata using configuration
    
    Args:
        row: Data row
        config: Dataset configuration from create_dataset_config
    
    Returns:
        Dictionary with instruction, response, and input
    """
    great_format = GReaTFormatHandler.create_great_format(
        row, 
        config['column_mapping'], 
        config['precision_mapping']
    )
    
    # Build rich prompt with metadata
    prompt_parts = []
    
    # Dataset description
    prompt_parts.append(f"Dataset: {config['dataset_name']}")
    prompt_parts.append(f"Description: {config['dataset_description']}")
    
    if config['domain_context']:
        prompt_parts.append(f"Context: {config['domain_context']}")
    
    # Column explanations
    if config['column_descriptions']:
        prompt_parts.append("\nColumn Explanations:")
        for col in config['columns']:
            display_name = config['column_mapping'].get(col, col)
            if col in config['column_descriptions']:
                prompt_parts.append(f"- {display_name}: {config['column_descriptions'][col]}")
            elif display_name in config['column_descriptions']:
                prompt_parts.append(f"- {display_name}: {config['column_descriptions'][display_name]}")
    
    # Example and generation request
    prompt_parts.append(f"\nExample {config['dataset_name']} record:")
    prompt_parts.append(great_format)
    prompt_parts.append(f"\nGenerate a new realistic {config['dataset_name']} record following the same format and column meanings:")
    
    full_prompt = "\n".join(prompt_parts)
    
    return {
        "instruction": full_prompt,
        "input": great_format,
        "response": great_format  # Expected output format
    }

def parse_generated_record(text: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse generated record using configuration
    
    Args:
        text: Generated text
        config: Dataset configuration
    
    Returns:
        Parsed record dictionary or None
    """
    return GReaTFormatHandler.parse_great_format(
        text,
        config['columns'],
        config['column_mapping'],
        config['type_mapping']
    )

def post_process_llm_response(llm_response: str, original_row: Union[pd.Series, Dict], 
                             config: Dict[str, Any]) -> Optional[pd.Series]:
    """
    Post-process LLM response to match exact tabular format of original row
    
    Args:
        llm_response: Raw response from LLM
        original_row: Original row to match format
        config: Dataset configuration
    
    Returns:
        pandas Series with same structure as original_row, or None if parsing fails
    """
    # Parse the LLM response
    parsed_dict = parse_generated_record(llm_response, config)
    
    if parsed_dict is None:
        return None
    
    # Convert to pandas Series with same structure as original
    if isinstance(original_row, pd.Series):
        # Create Series with same index as original
        result_series = pd.Series(index=original_row.index, dtype=object)
        
        # Map parsed values back to original column names
        reverse_mapping = {v: k for k, v in config['column_mapping'].items()}
        
        for col in original_row.index:
            # Check if column was mapped
            display_name = config['column_mapping'].get(col, col)
            
            if display_name in parsed_dict:
                result_series[col] = parsed_dict[display_name]
            elif col in parsed_dict:
                result_series[col] = parsed_dict[col]
            else:
                # Column not found in parsed response
                return None
        
        return result_series
    
    else:
        # Handle dictionary input
        result_dict = {}
        reverse_mapping = {v: k for k, v in config['column_mapping'].items()}
        
        for col in original_row.keys():
            display_name = config['column_mapping'].get(col, col)
            
            if display_name in parsed_dict:
                result_dict[col] = parsed_dict[display_name]
            elif col in parsed_dict:
                result_dict[col] = parsed_dict[col]
            else:
                return None
        
        return pd.Series(result_dict)

def validate_generated_row(generated_row: pd.Series, original_row: pd.Series, 
                          config: Dict[str, Any], tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate that generated row matches the expected format and constraints
    
    Args:
        generated_row: Generated row from LLM
        original_row: Original row for reference
        config: Dataset configuration
        tolerance: Tolerance for numerical validation
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'column_matches': {},
        'type_matches': {},
        'value_ranges': {}
    }
    
    # Check column structure
    if not generated_row.index.equals(original_row.index):
        validation_results['valid'] = False
        validation_results['errors'].append("Column structure mismatch")
        return validation_results
    
    # Check each column
    for col in generated_row.index:
        original_val = original_row[col]
        generated_val = generated_row[col]
        
        # Check if column exists
        validation_results['column_matches'][col] = generated_val is not None
        
        if generated_val is None:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing value for column: {col}")
            continue
        
        # Check type matching
        expected_type = config['type_mapping'].get(col, type(original_val))
        actual_type = type(generated_val)
        
        type_match = actual_type == expected_type or (
            # Allow int/float flexibility
            (expected_type in [int, float] and actual_type in [int, float])
        )
        
        validation_results['type_matches'][col] = type_match
        
        if not type_match:
            validation_results['warnings'].append(
                f"Type mismatch for {col}: expected {expected_type.__name__}, got {actual_type.__name__}"
            )
        
        # Check value ranges for numerical columns
        if isinstance(generated_val, (int, float)) and isinstance(original_val, (int, float)):
            # Simple range check based on original value
            if original_val != 0:
                relative_diff = abs(generated_val - original_val) / abs(original_val)
                validation_results['value_ranges'][col] = {
                    'original': original_val,
                    'generated': generated_val,
                    'relative_diff': relative_diff,
                    'within_tolerance': relative_diff <= tolerance * 10  # More lenient for synthetic data
                }
    
    return validation_results

def create_tabular_post_processor(config: Dict[str, Any]):
    """
    Create a post-processor function for a specific dataset configuration
    
    Args:
        config: Dataset configuration
    
    Returns:
        Function that can post-process LLM responses for this dataset
    """
    def post_processor(llm_response: str, original_row: Union[pd.Series, Dict]) -> Optional[pd.Series]:
        return post_process_llm_response(llm_response, original_row, config)
    
    return post_processor
