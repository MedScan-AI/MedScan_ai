"""
Baseline Synthetic Patient Data Generator using Faker

This module generates synthetic patient metadata for preprocessed medical images
using the Faker library for realistic patient demographics, symptoms, medications,
and examination details for tuberculosis chest X-ray and lung cancer CT scan datasets.
"""

import os
import csv
import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from glob import glob
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatientDataGenerator:
    """Base class for generating synthetic patient data using Faker."""
    
    def __init__(self, config: Dict):
        """
        Initialize the patient data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.demographics = config['demographics']
        self.general = config['general']
        
        # Initialize Faker
        faker_locale = config.get('faker', {}).get('locale', 'en_US')
        self.faker = Faker(faker_locale)
        
        # Set random seed for reproducibility
        seed = config['output'].get('random_seed')
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
            logger.info(f"Random seed set to {seed}")
    
    def extract_patient_id_from_filename(self, image_path: str) -> str:
        """
        Extract Patient ID from image filename.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Patient ID extracted from filename (without extension)
        """
        from pathlib import Path
        # Get filename without extension
        filename = Path(image_path).stem
        return filename
    
    def generate_patient_name(self) -> str:
        """
        Generate a random patient name using Faker.
        
        Returns:
            Full name string
        """
        return self.faker.name()
    
    def generate_age(self) -> int:
        """
        Generate a random age within configured range.
        
        Returns:
            Age in years
        """
        age_range = self.demographics['age_range']
        return self.faker.random_int(min=age_range['min'], max=age_range['max'])
    
    def generate_weight(self) -> float:
        """
        Generate a random weight within configured range using normal distribution.
        
        Returns:
            Weight in kg
        """
        weight_range = self.demographics['weight_range']
        # Use normal distribution for more realistic weights
        mean = (weight_range['min'] + weight_range['max']) / 2
        std = (weight_range['max'] - weight_range['min']) / 6
        
        # Generate weight with normal distribution
        weight = random.gauss(mean, std)
        weight = max(weight_range['min'], min(weight_range['max'], weight))
        return round(weight, 1)
    
    def generate_height(self) -> int:
        """
        Generate a random height within configured range using normal distribution.
        
        Returns:
            Height in cm
        """
        height_range = self.demographics['height_range']
        # Use normal distribution for more realistic heights
        mean = (height_range['min'] + height_range['max']) / 2
        std = (height_range['max'] - height_range['min']) / 6
        
        # Generate height with normal distribution
        height = random.gauss(mean, std)
        height = max(height_range['min'], min(height_range['max'], height))
        return round(height)
    
    def generate_gender(self) -> str:
        """
        Generate a random gender based on configured distribution.
        
        Returns:
            Gender string
        """
        gender_dist = self.demographics['gender_distribution']
        genders = list(gender_dist.keys())
        weights = list(gender_dist.values())
        return random.choices(genders, weights=weights)[0]
    
    def generate_symptoms(
        self, 
        symptom_list: List[str], 
        max_symptoms: Optional[int] = None
    ) -> str:
        """
        Generate presenting symptoms from available list.
        
        Args:
            symptom_list: List of possible symptoms
            max_symptoms: Maximum number of symptoms to combine
            
        Returns:
            Comma-separated symptom string
        """
        if max_symptoms is None:
            max_symptoms = self.general['max_symptoms_per_patient']
        
        num_symptoms = random.randint(1, min(max_symptoms, len(symptom_list)))
        symptoms = random.sample(symptom_list, num_symptoms)
        return ", ".join(symptoms)
    
    def generate_medications(self, medication_list: List[str]) -> str:
        """
        Generate current medications from available list.
        
        Args:
            medication_list: List of possible medications
            
        Returns:
            Comma-separated medication string
        """
        # Probability of no medications
        if random.random() < self.general['prob_no_medication']:
            return "None"
        
        max_meds = self.general['max_medications_per_patient']
        # Filter out "None" from the list for selection
        available_meds = [m for m in medication_list if m != "None"]
        
        if not available_meds:
            return "None"
        
        num_meds = random.randint(1, min(max_meds, len(available_meds)))
        medications = random.sample(available_meds, num_meds)
        return ", ".join(medications)
    
    def generate_surgeries(self, surgery_list: List[str]) -> str:
        """
        Generate previous surgeries from available list.
        
        Args:
            surgery_list: List of possible surgeries
            
        Returns:
            Comma-separated surgery string
        """
        # Probability of no surgeries
        if random.random() < self.general['prob_no_surgery']:
            return "None"
        
        max_surgeries = self.general['max_surgeries_per_patient']
        # Filter out "None" from the list for selection
        available_surgeries = [s for s in surgery_list if s != "None"]
        
        if not available_surgeries:
            return "None"
        
        num_surgeries = random.randint(1, min(max_surgeries, len(available_surgeries)))
        surgeries = random.sample(available_surgeries, num_surgeries)
        return ", ".join(surgeries)
    
    def generate_urgency(self, urgency_levels: Dict[str, float]) -> str:
        """
        Generate urgency level based on configured distribution.
        
        Args:
            urgency_levels: Dictionary mapping urgency levels to probability weights
            
        Returns:
            Urgency level string
        """
        levels = list(urgency_levels.keys())
        weights = list(urgency_levels.values())
        return random.choices(levels, weights=weights)[0]


class TBDataGenerator(PatientDataGenerator):
    """Generator for tuberculosis chest X-ray patient data using Faker."""
    
    def __init__(self, config: Dict):
        """
        Initialize the tuberculosis data generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.tb_config = config['tb']
    
    def generate_patient_record(
        self, 
        image_path: str, 
        class_name: str
    ) -> Dict[str, str]:
        """
        Generate a complete patient record for a tuberculosis chest X-ray image.
        
        Args:
            image_path: Path to the image file
            class_name: Class (Normal, Tuberculosis)
            
        Returns:
            Dictionary with patient data
        """
        # Extract Patient ID from filename
        patient_id = self.extract_patient_id_from_filename(image_path)
        
        # Get class-specific symptoms
        symptoms_list = self.tb_config['presenting_symptoms'].get(
            class_name, 
            self.tb_config['presenting_symptoms']['Normal']
        )
        
        record = {
            'Patient_Full_Name': self.generate_patient_name(),
            'Patient_ID': patient_id,
            'Presenting_Symptoms': self.generate_symptoms(symptoms_list),
            'Current_Medications': self.generate_medications(
                self.tb_config['current_medications']
            ),
            'Previous_Relevant_Surgeries': self.generate_surgeries(
                self.tb_config['previous_surgeries']
            ),
            'Age_Years': self.generate_age(),
            'Weight_KG': self.generate_weight(),
            'Height_CM': self.generate_height(),
            'Gender': self.generate_gender(),
            'Examination_Type': self.tb_config['examination_type'],
            'Body_Region': self.tb_config['body_region'],
            'Urgency_Level': self.generate_urgency(
                self.tb_config['urgency_levels']
            )
        }
        
        # Add image path if configured
        if self.config['output'].get('include_image_path', True):
            record['Image_Path'] = image_path
            record['Diagnosis_Class'] = class_name
        
        return record


class LungCancerDataGenerator(PatientDataGenerator):
    """Generator for lung cancer CT scan patient data using Faker."""
    
    def __init__(self, config: Dict):
        """
        Initialize the lung cancer data generator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.lung_config = config['lung_cancer']
    
    def generate_patient_record(
        self, 
        image_path: str, 
        class_name: str
    ) -> Dict[str, str]:
        """
        Generate a complete patient record for a lung cancer CT scan image.
        
        Args:
            image_path: Path to the image file
            class_name: Cancer class (adenocarcinoma, etc.)
            
        Returns:
            Dictionary with patient data
        """
        # Extract Patient ID from filename
        patient_id = self.extract_patient_id_from_filename(image_path)
        
        # Get class-specific symptoms
        symptoms_list = self.lung_config['presenting_symptoms'].get(
            class_name,
            self.lung_config['presenting_symptoms']['normal']
        )
        
        record = {
            'Patient_Full_Name': self.generate_patient_name(),
            'Patient_ID': patient_id,
            'Presenting_Symptoms': self.generate_symptoms(symptoms_list),
            'Current_Medications': self.generate_medications(
                self.lung_config['current_medications']
            ),
            'Previous_Relevant_Surgeries': self.generate_surgeries(
                self.lung_config['previous_surgeries']
            ),
            'Age_Years': self.generate_age(),
            'Weight_KG': self.generate_weight(),
            'Height_CM': self.generate_height(),
            'Gender': self.generate_gender(),
            'Examination_Type': self.lung_config['examination_type'],
            'Body_Region': self.lung_config['body_region'],
            'Urgency_Level': self.generate_urgency(
                self.lung_config['urgency_levels']
            )
        }
        
        # Add image path if configured
        if self.config['output'].get('include_image_path', True):
            record['Image_Path'] = image_path
            record['Diagnosis_Class'] = class_name
        
        return record


class SyntheticDataPipeline:
    """Main pipeline for generating synthetic patient data using Faker."""
    
    def __init__(self, config_path: str):
        """
        Initialize the synthetic data pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize generators
        self.tb_generator = TBDataGenerator(self.config)
        self.lung_generator = LungCancerDataGenerator(self.config)
    
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        logger.info(f"Loading configuration from: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _discover_partitions(self, base_path: str) -> List[Dict]:
        """
        Discover all year/month/day partitions in a base path.
        
        Args:
            base_path: Base directory to search
            
        Returns:
            List of partition info dicts with path and timestamp
        """
        partitions = []
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            return partitions
        
        # Find all YYYY/MM/DD directories
        pattern = str(base_path_obj / "[0-9][0-9][0-9][0-9]" / "[0-9][0-9]" / "[0-9][0-9]")
        partition_dirs = glob(pattern)
        
        for partition_dir in sorted(partition_dirs):
            # Extract year/month/day from path
            parts = Path(partition_dir).parts[-3:]
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                timestamp = datetime(year, month, day)
                
                partitions.append({
                    'path': Path(partition_dir),
                    'timestamp': timestamp.isoformat(),
                    'date_obj': timestamp
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse partition path {partition_dir}: {e}")
                continue
        
        return partitions
    
    def _get_partition_path(self, base_path: str, timestamp: datetime = None) -> Path:
        """
        Generate partition path based on year/month/day structure.
        
        Args:
            base_path: Base directory path
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Full partition path: base_path/YYYY/MM/DD
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        partition_path = Path(base_path) / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        return partition_path
    
    def get_image_files(self, preprocessed_path: str, use_latest_partition: bool = True) -> List[Tuple[str, str, str]]:
        """
        Get all preprocessed image files with their split and class.
        Supports partitioned data (YYYY/MM/DD) and will use latest partition by default.
        
        Args:
            preprocessed_path: Path to preprocessed data directory
            use_latest_partition: If True, only use latest partition; if False, use all partitions
            
        Returns:
            List of tuples (image_path, split, class_name)
        """
        image_files = []
        preprocessed_dir = Path(preprocessed_path)
        
        if not preprocessed_dir.exists():
            logger.warning(f"Preprocessed directory not found: {preprocessed_dir}")
            return image_files
        
        # Check if data is partitioned
        partitions = self._discover_partitions(str(preprocessed_dir))
        
        if partitions:
            if use_latest_partition:
                # Use only the latest partition
                latest_partition = partitions[-1]
                logger.info(f"Using latest partition: {latest_partition['timestamp']}")
                logger.info(f"  Path: {latest_partition['path']}")
                dirs_to_process = [latest_partition['path']]
            else:
                # Use all partitions
                logger.info(f"Found {len(partitions)} partition(s), processing all")
                dirs_to_process = [p['path'] for p in partitions]
        else:
            # Non-partitioned data (fallback)
            logger.info("No partitions found, using base directory")
            dirs_to_process = [preprocessed_dir]
        
        # Iterate through selected directories
        for base_dir in dirs_to_process:
            # Iterate through splits and classes
            for split_dir in base_dir.iterdir():
                if split_dir.is_dir():
                    split_name = split_dir.name
                    
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            class_name = class_dir.name
                            
                            # Get all image files
                            for img_path in class_dir.glob('*.jpg'):
                                # Store path as string (absolute or relative)
                                # Use absolute path if possible, otherwise use as-is
                                try:
                                    if img_path.is_absolute():
                                        rel_path = str(img_path.relative_to(Path.cwd()))
                                    else:
                                        rel_path = str(img_path)
                                except ValueError:
                                    # If relative_to fails, just use the path as-is
                                    rel_path = str(img_path)
                                image_files.append((rel_path, split_name, class_name))
        
        return image_files
    
    def generate_tb_data(self) -> List[Dict[str, str]]:
        """
        Generate synthetic patient data for all tuberculosis chest X-ray images.
        
        Returns:
            List of patient records
        """
        logger.info("Generating synthetic data for tuberculosis chest X-rays...")
        
        # Get preprocessed images
        preprocessed_path = "data/preprocessed/tb"
        image_files = self.get_image_files(preprocessed_path)
        
        if not image_files:
            logger.warning("No tuberculosis chest X-ray images found!")
            return []
        
        logger.info(f"Found {len(image_files)} tuberculosis chest X-ray images")
        
        # Generate patient records
        records = []
        for idx, (img_path, split, class_name) in enumerate(image_files, start=1):
            record = self.tb_generator.generate_patient_record(
                image_path=img_path,
                class_name=class_name
            )
            records.append(record)
            
            if idx % 500 == 0:
                logger.info(f"Generated {idx}/{len(image_files)} records...")
        
        logger.info(f"Generated {len(records)} tuberculosis patient records")
        return records
    
    def generate_lung_cancer_data(self) -> List[Dict[str, str]]:
        """
        Generate synthetic patient data for all lung cancer CT scan images.
        
        Returns:
            List of patient records
        """
        logger.info("Generating synthetic data for lung cancer CT scans...")
        
        # Get preprocessed images
        preprocessed_path = "data/preprocessed/lung_cancer_ct_scan"
        image_files = self.get_image_files(preprocessed_path)
        
        if not image_files:
            logger.warning("No lung cancer CT scan images found!")
            return []
        
        logger.info(f"Found {len(image_files)} lung cancer CT scan images")
        
        # Generate patient records
        records = []
        for idx, (img_path, split, class_name) in enumerate(image_files, start=1):
            record = self.lung_generator.generate_patient_record(
                image_path=img_path,
                class_name=class_name
            )
            records.append(record)
            
            if idx % 500 == 0:
                logger.info(f"Generated {idx}/{len(image_files)} records...")
        
        logger.info(f"Generated {len(records)} lung cancer patient records")
        return records
    
    def save_records_csv(self, records: List[Dict[str, str]], output_path: str, use_partitioning: bool = True) -> None:
        """
        Save patient records to CSV file with optional year/month/day partitioning.
        
        Args:
            records: List of patient records
            output_path: Base path to output CSV file
            use_partitioning: If True, save to YYYY/MM/DD partition
        """
        if not records:
            logger.warning("No records to save!")
            return
        
        # Determine final output path
        if use_partitioning:
            # Extract base path and filename
            output_file = Path(output_path)
            base_dir = output_file.parent
            filename = output_file.name
            
            # Get partition path for today
            partition_dir = self._get_partition_path(str(base_dir))
            final_output_path = partition_dir / filename
            
            logger.info(f"Using partitioned output: {partition_dir}")
        else:
            final_output_path = Path(output_path)
        
        # Create output directory
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        fieldnames = records[0].keys()
        
        with open(final_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        logger.info(f"Saved {len(records)} records to {final_output_path}")
    
    def save_records_json(self, records: List[Dict[str, str]], output_path: str, use_partitioning: bool = True) -> None:
        """
        Save patient records to JSON file with optional year/month/day partitioning.
        
        Args:
            records: List of patient records
            output_path: Base path to output JSON file
            use_partitioning: If True, save to YYYY/MM/DD partition
        """
        if not records:
            logger.warning("No records to save!")
            return
        
        # Determine final output path
        if use_partitioning:
            # Extract base path and filename
            output_file = Path(output_path)
            base_dir = output_file.parent
            filename = output_file.name
            
            # Get partition path for today
            partition_dir = self._get_partition_path(str(base_dir))
            final_output_path = partition_dir / filename
            
            logger.info(f"Using partitioned output: {partition_dir}")
        else:
            final_output_path = Path(output_path)
        
        # Create output directory
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(records)} records to {final_output_path}")
    
    def run(self) -> None:
        """Run the complete synthetic data generation pipeline with partition support."""
        logger.info("Starting synthetic patient data generation pipeline...")
        logger.info(f"Using Faker library for realistic data generation")
        
        output_format = self.config['output'].get('format', 'csv')
        use_partitioning = self.config['output'].get('use_partitioning', True)
        use_latest_partition = self.config['output'].get('use_latest_partition', True)
        
        logger.info(f"Partitioning enabled: {use_partitioning}")
        logger.info(f"Using latest partition only: {use_latest_partition}")
        
        # Generate tuberculosis data (for latest partition only)
        tb_records = self.generate_tb_data()
        if tb_records:
            tb_output_path = self.config['output']['tb_output_path']
            
            if output_format == 'csv':
                self.save_records_csv(tb_records, tb_output_path, use_partitioning=use_partitioning)
            else:
                # Change extension to .json if needed
                tb_output_path = tb_output_path.replace('.csv', '.json')
                self.save_records_json(tb_records, tb_output_path, use_partitioning=use_partitioning)
        
        # Generate lung cancer data (for latest partition only)
        lung_records = self.generate_lung_cancer_data()
        if lung_records:
            lung_output_path = self.config['output']['lung_cancer_output_path']
            
            if output_format == 'csv':
                self.save_records_csv(lung_records, lung_output_path, use_partitioning=use_partitioning)
            else:
                # Change extension to .json if needed
                lung_output_path = lung_output_path.replace('.csv', '.json')
                self.save_records_json(lung_records, lung_output_path, use_partitioning=use_partitioning)
        
        logger.info("Synthetic patient data generation complete!")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("="*60)
        print(f"Tuberculosis Chest X-ray Records: {len(tb_records)}")
        print(f"Lung Cancer CT Scan Records: {len(lung_records)}")
        print(f"Total Records Generated: {len(tb_records) + len(lung_records)}")
        print(f"Output Format: {output_format.upper()}")
        print(f"Partitioning Enabled: {use_partitioning}")
        print(f"Latest Partition Only: {use_latest_partition}")
        print(f"Faker Locale: {self.config.get('faker', {}).get('locale', 'en_US')}")
        if use_partitioning:
            current_date = datetime.now()
            partition_path = f"{current_date.year:04d}/{current_date.month:02d}/{current_date.day:02d}"
            print(f"Output Partition: {partition_path}")
        print("="*60)


def main():
    """Main entry point for synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic patient metadata for preprocessed medical images using Faker'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/synthetic_data.yml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = SyntheticDataPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
