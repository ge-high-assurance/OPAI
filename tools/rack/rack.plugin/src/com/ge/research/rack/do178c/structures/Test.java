/*
 * BSD 3-Clause License
 * 
 * Copyright (c) 2023, General Electric Company and Galois, Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.ge.research.rack.do178c.structures;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Saswata Paul
 */
public class Test {

    private String id;
    private String result;
    private String type;

    private List<String> verifies = new ArrayList<String>();
    private List<ReviewLog> logs = new ArrayList<ReviewLog>();

    private List<String> sourceDocument = new ArrayList<String>();

    /**
     * @return the sourceDocument
     */
    public List<String> getSourceDocument() {
        return sourceDocument;
    }

    /**
     * @param sourceDocument the sourceDocument to set
     */
    public void setSourceDocument(List<String> sourceDocument) {
        this.sourceDocument = sourceDocument;
    }

    /**
     * @param id
     * @param result
     * @param aVerifies
     */
    public Test(String id, String result, String aVerifies, String type) {
        super();
        this.id = id;
        this.result = result;
        this.verifies.add(aVerifies);
        this.type = type;
    }

    /**
     * @return the logs
     */
    public List<ReviewLog> getLogs() {
        return logs;
    }
    /**
     * @param logs the logs to set
     */
    public void setLogs(List<ReviewLog> logs) {
        this.logs = logs;
    }

    /**
     * @return the id
     */
    public String getId() {
        return id;
    }
    /**
     * @param id the id to set
     */
    public void setId(String id) {
        this.id = id;
    }
    /**
     * @return the result
     */
    public String getResult() {
        return result;
    }
    /**
     * @param result the result to set
     */
    public void setResult(String result) {
        this.result = result;
    }

    /**
     * @return the verifies
     */
    public List<String> getVerifies() {
        return verifies;
    }
    /**
     * @param verifies the verifies to set
     */
    public void setVerifies(List<String> verifies) {
        this.verifies = verifies;
    }
    /**
     * @return the type
     */
    public String getType() {
        return type;
    }
    /**
     * @param type the type to set
     */
    public void setType(String type) {
        this.type = type;
    }

    /**
     * Creates a string with all verifies names
     *
     * @return
     */
    public String getVerifiesAsString() {
        String list = "";
        if (this.verifies != null) {
            if (this.verifies.size() > 0) {
                for (String req : this.verifies) {
                    list = list + req + ", ";
                }
            }
        }
        return list;
    }
}
