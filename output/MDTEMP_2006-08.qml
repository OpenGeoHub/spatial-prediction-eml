<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" minScale="1e+8" maxScale="0" version="3.18.0-Zürich" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer classificationMin="143" band="1" opacity="1" nodataColor="" type="singlebandpseudocolor" classificationMax="239" alphaBand="-1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>CumulativeCut</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="239" clip="0" minimumValue="143" colorRampType="INTERPOLATED" classificationMode="1" labelPrecision="0">
          <colorramp type="cpt-city" name="[source]">
            <Option type="Map">
              <Option type="QString" value="0" name="inverted"/>
              <Option type="QString" value="cpt-city" name="rampType"/>
              <Option type="QString" value="jjg/misc/temperature" name="schemeName"/>
              <Option type="QString" value="" name="variantName"/>
            </Option>
            <prop k="inverted" v="0"/>
            <prop k="rampType" v="cpt-city"/>
            <prop k="schemeName" v="jjg/misc/temperature"/>
            <prop k="variantName" v=""/>
          </colorramp>
          <item value="43" color="#1316b4" label="43" alpha="255"/>
          <item value="53.092" color="#2331c7" label="53" alpha="255"/>
          <item value="63.184" color="#2d42c9" label="63" alpha="255"/>
          <item value="73.2528" color="#3755cb" label="73" alpha="255"/>
          <item value="83.34479999999999" color="#365fc6" label="83" alpha="255"/>
          <item value="93.4368" color="#466fcf" label="93" alpha="255"/>
          <item value="103.5288" color="#507dd2" label="104" alpha="255"/>
          <item value="113.5976" color="#598dd6" label="114" alpha="255"/>
          <item value="123.6896" color="#629bd9" label="124" alpha="255"/>
          <item value="133.7816" color="#7eb9e9" label="134" alpha="255"/>
          <item value="143.8736" color="#a5d7ff" label="144" alpha="255"/>
          <item value="153.9656" color="#c4e5b7" label="154" alpha="255"/>
          <item value="164.0344" color="#b4dfa8" label="164" alpha="255"/>
          <item value="174.12640000000002" color="#b0d793" label="174" alpha="255"/>
          <item value="184.2184" color="#c7cf74" label="184" alpha="255"/>
          <item value="194.3104" color="#dbc85b" label="194" alpha="255"/>
          <item value="204.4024" color="#debd50" label="204" alpha="255"/>
          <item value="214.47119999999998" color="#d9a449" label="214" alpha="255"/>
          <item value="224.5632" color="#d39242" label="225" alpha="255"/>
          <item value="234.65519999999998" color="#d1853e" label="235" alpha="255"/>
          <item value="244.74720000000002" color="#cc7139" label="245" alpha="255"/>
          <item value="254.816" color="#ca6232" label="255" alpha="255"/>
          <item value="264.908" color="#c74528" label="265" alpha="255"/>
          <item value="275" color="#c74528" label="275" alpha="255"/>
          <rampLegendSettings minimumLabel="" suffix="" orientation="2" maximumLabel="" prefix="" direction="0">
            <numericFormat id="basic">
              <Option type="Map">
                <Option type="QChar" value="" name="decimal_separator"/>
                <Option type="int" value="6" name="decimals"/>
                <Option type="int" value="0" name="rounding_type"/>
                <Option type="bool" value="false" name="show_plus"/>
                <Option type="bool" value="true" name="show_thousand_separator"/>
                <Option type="bool" value="false" name="show_trailing_zeros"/>
                <Option type="QChar" value="" name="thousand_separator"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" gamma="1" contrast="0"/>
    <huesaturation colorizeBlue="128" colorizeOn="0" saturation="0" colorizeStrength="100" grayscaleMode="0" colorizeGreen="128" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
