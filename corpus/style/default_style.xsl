<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
		xmlns:mmax="org.eml.MMAX2.discourse.MMAX2DiscourseLoader"
		xmlns:sentiment="www.eml.org/NameSpaces/sentiment">
  <xsl:output method="text" indent="no" omit-xml-declaration="yes"/>
  <xsl:strip-space elements="*"/>

  <xsl:template match="words">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="word">
    <xsl:variable name="text" select="." />
    <xsl:choose>
      <xsl:when test="$text = 'EOL'">
	<xsl:text>&#xA;&#xA;</xsl:text>
      </xsl:when>

      <xsl:when test="1">
	<xsl:value-of select="mmax:registerDiscourseElement(@id)"/>

	<xsl:value-of select="mmax:setDiscourseElementStart()"/>
	<xsl:apply-templates/>
	<xsl:value-of select="mmax:setDiscourseElementEnd()"/>
	<xsl:text> </xsl:text>
      </xsl:when>
    </xsl:choose>
  </xsl:template>
</xsl:stylesheet>
