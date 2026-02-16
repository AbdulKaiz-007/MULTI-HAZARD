"""
Cascading Risk Logic Module
Rule-based risk assessment for multi-hazard scenarios
"""

import numpy as np
from typing import Dict, List, Tuple


class CascadingRiskAnalyzer:
    """
    Analyzes cascading risks based on detected hazards
    
    Risk Categories:
        - Infrastructure Risk
        - Landslide Risk
        - Flood Propagation Risk
        - Fire Spread Risk
        - Compound Disaster Risk
    """
    
    def __init__(self, image_size=(256, 256)):
        """
        Args:
            image_size (tuple): Size of the analyzed image
        """
        self.image_size = image_size
        self.total_pixels = image_size[0] * image_size[1]
        
        # Risk thresholds (percentage of image area)
        self.thresholds = {
            'flood_low': 0.05,      # 5% of image
            'flood_medium': 0.15,   # 15% of image
            'flood_high': 0.30,     # 30% of image
            'fire_low': 0.03,       # 3% of image
            'fire_medium': 0.10,    # 10% of image
            'fire_high': 0.25,      # 25% of image
            'damage_low': 0.05,     # 5% of image
            'damage_medium': 0.15,  # 15% of image
            'damage_high': 0.30     # 30% of image
        }
    
    def calculate_hazard_areas(self, prediction_mask):
        """
        Calculate area coverage for each hazard type
        
        Args:
            prediction_mask (np.ndarray): Predicted segmentation mask [H, W]
        
        Returns:
            areas (dict): Dictionary with area percentages for each class
        """
        areas = {
            'background': 0.0,
            'flood': 0.0,
            'fire': 0.0,
            'damage': 0.0
        }
        
        class_names = ['background', 'flood', 'fire', 'damage']
        
        for class_idx, class_name in enumerate(class_names):
            class_pixels = np.sum(prediction_mask == class_idx)
            percentage = (class_pixels / self.total_pixels) * 100
            areas[class_name] = percentage
        
        return areas
    
    def assess_infrastructure_risk(self, areas):
        """
        Assess infrastructure risk based on flood and damage
        
        Logic:
            - High flood + High damage = Critical Infrastructure Risk
            - Medium flood + Medium damage = High Infrastructure Risk
            - Low flood + Low damage = Moderate Infrastructure Risk
        
        Args:
            areas (dict): Hazard area percentages
        
        Returns:
            risk_level (str): Risk level
            risk_score (float): Risk score (0-100)
        """
        flood_pct = areas['flood']
        damage_pct = areas['damage']
        
        # Calculate combined risk score
        risk_score = (flood_pct * 0.6 + damage_pct * 0.4)
        
        if flood_pct > self.thresholds['flood_high'] * 100 and \
           damage_pct > self.thresholds['damage_high'] * 100:
            return "Critical Infrastructure Risk", min(risk_score, 100)
        elif flood_pct > self.thresholds['flood_medium'] * 100 and \
             damage_pct > self.thresholds['damage_medium'] * 100:
            return "High Infrastructure Risk", min(risk_score, 100)
        elif flood_pct > self.thresholds['flood_low'] * 100 or \
             damage_pct > self.thresholds['damage_low'] * 100:
            return "Moderate Infrastructure Risk", min(risk_score, 100)
        else:
            return "Low Infrastructure Risk", min(risk_score, 100)
    
    def assess_landslide_risk(self, areas, rainfall_intensity=None):
        """
        Assess landslide risk based on flood and fire
        
        Logic:
            - High flood + Fire presence = High Landslide Risk
              (fire weakens soil, flood triggers slides)
        
        Args:
            areas (dict): Hazard area percentages
            rainfall_intensity (float): Optional rainfall data
        
        Returns:
            risk_level (str): Risk level
            risk_score (float): Risk score (0-100)
        """
        flood_pct = areas['flood']
        fire_pct = areas['fire']
        
        # Base risk from flood
        risk_score = flood_pct * 0.7
        
        # Fire increases risk (soil destabilization)
        if fire_pct > self.thresholds['fire_low'] * 100:
            risk_score += fire_pct * 0.3
        
        # Rainfall intensity modifier (if available)
        if rainfall_intensity is not None:
            risk_score *= (1 + rainfall_intensity / 100)
        
        risk_score = min(risk_score, 100)
        
        if risk_score > 60:
            return "High Landslide Risk", risk_score
        elif risk_score > 30:
            return "Moderate Landslide Risk", risk_score
        elif risk_score > 10:
            return "Low Landslide Risk", risk_score
        else:
            return "Minimal Landslide Risk", risk_score
    
    def assess_fire_spread_risk(self, areas):
        """
        Assess fire spread risk
        
        Logic:
            - High fire area = High spread risk
            - Fire + Damage = Increased spread (damaged structures fuel fire)
        
        Args:
            areas (dict): Hazard area percentages
        
        Returns:
            risk_level (str): Risk level
            risk_score (float): Risk score (0-100)
        """
        fire_pct = areas['fire']
        damage_pct = areas['damage']
        
        # Base fire risk
        risk_score = fire_pct
        
        # Damaged buildings increase spread risk
        if damage_pct > self.thresholds['damage_low'] * 100:
            risk_score += damage_pct * 0.3
        
        risk_score = min(risk_score, 100)
        
        if risk_score > 50:
            return "Critical Fire Spread Risk", risk_score
        elif risk_score > 25:
            return "High Fire Spread Risk", risk_score
        elif risk_score > 10:
            return "Moderate Fire Spread Risk", risk_score
        else:
            return "Low Fire Spread Risk", risk_score
    
    def assess_compound_disaster_risk(self, areas):
        """
        Assess compound disaster risk (multiple hazards present)
        
        Logic:
            - Multiple hazards present = Compound disaster
            - Higher combined area = Higher risk
        
        Args:
            areas (dict): Hazard area percentages
        
        Returns:
            risk_level (str): Risk level
            risk_score (float): Risk score (0-100)
            active_hazards (list): List of active hazards
        """
        active_hazards = []
        
        if areas['flood'] > self.thresholds['flood_low'] * 100:
            active_hazards.append('Flood')
        if areas['fire'] > self.thresholds['fire_low'] * 100:
            active_hazards.append('Fire')
        if areas['damage'] > self.thresholds['damage_low'] * 100:
            active_hazards.append('Building Damage')
        
        num_hazards = len(active_hazards)
        
        # Calculate compound risk
        total_hazard_area = areas['flood'] + areas['fire'] + areas['damage']
        risk_score = total_hazard_area * (1 + 0.3 * num_hazards)
        risk_score = min(risk_score, 100)
        
        if num_hazards >= 3:
            return "Critical Compound Disaster", risk_score, active_hazards
        elif num_hazards == 2:
            return "High Compound Risk", risk_score, active_hazards
        elif num_hazards == 1:
            return "Single Hazard Event", risk_score, active_hazards
        else:
            return "No Significant Hazard", risk_score, active_hazards
    
    def analyze(self, prediction_mask, rainfall_intensity=None):
        """
        Perform complete cascading risk analysis
        
        Args:
            prediction_mask (np.ndarray): Predicted segmentation mask
            rainfall_intensity (float): Optional rainfall data
        
        Returns:
            analysis (dict): Complete risk analysis report
        """
        # Calculate hazard areas
        areas = self.calculate_hazard_areas(prediction_mask)
        
        # Assess different risk types
        infra_risk, infra_score = self.assess_infrastructure_risk(areas)
        landslide_risk, landslide_score = self.assess_landslide_risk(
            areas, rainfall_intensity
        )
        fire_risk, fire_score = self.assess_fire_spread_risk(areas)
        compound_risk, compound_score, active_hazards = \
            self.assess_compound_disaster_risk(areas)
        
        # Compile analysis
        analysis = {
            'hazard_areas': areas,
            'infrastructure_risk': {
                'level': infra_risk,
                'score': infra_score
            },
            'landslide_risk': {
                'level': landslide_risk,
                'score': landslide_score
            },
            'fire_spread_risk': {
                'level': fire_risk,
                'score': fire_score
            },
            'compound_disaster': {
                'level': compound_risk,
                'score': compound_score,
                'active_hazards': active_hazards
            },
            'overall_risk_score': max(infra_score, landslide_score, 
                                     fire_score, compound_score)
        }
        
        return analysis
    
    def generate_report(self, analysis):
        """
        Generate human-readable risk report
        
        Args:
            analysis (dict): Risk analysis results
        
        Returns:
            report (str): Formatted report
        """
        report = []
        report.append("=" * 70)
        report.append("CASCADING RISK ANALYSIS REPORT")
        report.append("=" * 70)
        
        # Hazard areas
        report.append("\n📊 DETECTED HAZARD AREAS:")
        report.append("-" * 70)
        for hazard, percentage in analysis['hazard_areas'].items():
            if hazard != 'background':
                report.append(f"  {hazard.capitalize():15s}: {percentage:6.2f}%")
        
        # Infrastructure risk
        report.append("\n🏗️  INFRASTRUCTURE RISK:")
        report.append("-" * 70)
        report.append(f"  Level: {analysis['infrastructure_risk']['level']}")
        report.append(f"  Score: {analysis['infrastructure_risk']['score']:.2f}/100")
        
        # Landslide risk
        report.append("\n⛰️  LANDSLIDE RISK:")
        report.append("-" * 70)
        report.append(f"  Level: {analysis['landslide_risk']['level']}")
        report.append(f"  Score: {analysis['landslide_risk']['score']:.2f}/100")
        
        # Fire spread risk
        report.append("\n🔥 FIRE SPREAD RISK:")
        report.append("-" * 70)
        report.append(f"  Level: {analysis['fire_spread_risk']['level']}")
        report.append(f"  Score: {analysis['fire_spread_risk']['score']:.2f}/100")
        
        # Compound disaster
        report.append("\n⚠️  COMPOUND DISASTER ASSESSMENT:")
        report.append("-" * 70)
        report.append(f"  Level: {analysis['compound_disaster']['level']}")
        report.append(f"  Score: {analysis['compound_disaster']['score']:.2f}/100")
        report.append(f"  Active Hazards: {', '.join(analysis['compound_disaster']['active_hazards'])}")
        
        # Overall risk
        report.append("\n🎯 OVERALL RISK SCORE:")
        report.append("-" * 70)
        report.append(f"  {analysis['overall_risk_score']:.2f}/100")
        
        report.append("=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Cascading Risk Logic Module - Test")
    print("=" * 70)
    
    # Create sample prediction mask
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    test_mask[50:100, 50:150] = 1  # Flood
    test_mask[120:180, 80:200] = 2  # Fire
    test_mask[200:240, 100:180] = 3  # Damage
    
    # Create analyzer
    analyzer = CascadingRiskAnalyzer(image_size=(256, 256))
    
    # Perform analysis
    analysis = analyzer.analyze(test_mask, rainfall_intensity=50)
    
    # Generate report
    report = analyzer.generate_report(analysis)
    print(report)
